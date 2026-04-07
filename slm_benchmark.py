"""SLM SQL Benchmark - Core functions for testing Small Language Models on SQL generation."""

import subprocess
import socket
import os
import time
import json
import json as _json
import re
import datetime
import threading
import pathlib
import io
import zipfile

import requests
import duckdb
import pandas as pd
import numpy as np
import psutil
import yaml
from IPython.display import display

# ============================================
# Module-level state (set by init())
# ============================================
con = None
timestamp = None
questions = []
last_tested_model = None
llama_server_process = None

# ============================================
# Config defaults (overridden from notebook)
# ============================================
LLAMA_CPP_ENDPOINT      = "http://127.0.0.1:8080/v1/chat/completions"
TIMEOUT_SECONDS         = None
max_attempts            = None
SF                      = 1
output_dir              = r"c:\llm"
data_dir                = r"c:\llm\data"
test_semantic_model_url = r'c:\llm\semantic_model.txt'
questions_url           = r'c:\llm\questions\questions.json'
model0                  = "claude-opus-4-6"
enable_feedback_loop    = True
MODEL_CONFIGS           = {}
_token_accumulator      = {"completion_tokens": 0, "generation_ms": 0.0}


def configure(**kwargs):
    """Update module-level config from keyword arguments."""
    g = globals()
    for key, value in kwargs.items():
        if key in g:
            g[key] = value
        else:
            raise ValueError(f"Unknown config key: {key}")


def load_config(path=r'c:\llm\models_config.json'):
    """Load MODEL_CONFIGS and settings from a JSON config file."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    global MODEL_CONFIGS
    MODEL_CONFIGS = cfg.get('models', {})
    for key, value in cfg.get('settings', {}).items():
        configure(**{key: value})
    print(f"Loaded {len(MODEL_CONFIGS)} model(s) from {path}")
    init()


def _create_views(c):
    c.sql("""
        CREATE OR REPLACE VIEW v_transactions AS
        SELECT
            ss_store_sk                            AS store_sk,
            ss_item_sk                             AS item_sk,
            ss_customer_sk                         AS customer_sk,
            ss_sold_date_sk                        AS date_sk,
            ss_ticket_number                       AS ticket_number,
            COALESCE(ss_quantity, 0)               AS quantity,
            COALESCE(ss_sales_price * ss_quantity, 0.0) AS sale_amount,
            0.0                                    AS return_amount,
            'sale'                                 AS row_type
        FROM store_sales
        UNION ALL
        SELECT
            sr_store_sk                            AS store_sk,
            sr_item_sk                             AS item_sk,
            sr_customer_sk                         AS customer_sk,
            sr_returned_date_sk                    AS date_sk,
            sr_ticket_number                       AS ticket_number,
            0                                      AS quantity,
            0.0                                    AS sale_amount,
            COALESCE(sr_return_amt, 0.0)           AS return_amount,
            'return'                               AS row_type
        FROM store_returns
    """)


def init():
    """Initialize database connection, timestamp, and load questions."""
    global con, timestamp, questions, db_path

    if SF < 1:
        schema = f"DS{str(SF).replace('.', '_')}"
    else:
        schema = f'DS{SF:02d}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, f"{schema}.duckdb")

    if not pathlib.Path(db_path).exists():
        c = duckdb.connect(db_path)
        c.sql("SET memory_limit = '14GB' ")
        c.sql(f"CALL dsdgen(sf={SF})")
        _create_views(c)
        c.close()

    con = duckdb.connect(db_path, read_only=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load questions
    try:
        response = requests.get(questions_url)
        response.raise_for_status()
        questions = json.loads(response.text)
        print("Loaded questions from GitHub.")
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        try:
            with open(questions_url, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print("Loaded questions from local path.")
        except Exception as e2:
            print(f"Error loading questions: {e2}")
            questions = []

    print(f"Database: {db_path} | Questions: {len(questions)} | SF: {SF}")


# ============================================
# llama.cpp update
# ============================================
LLAMA_CPP_DIR = r'c:\llm\llama'
CUDA_VERSION = '12.4'
VERSION_FILE = os.path.join(LLAMA_CPP_DIR, 'version.txt')


def update_llama_cpp():
    """Download and install the latest llama.cpp CUDA release if newer version available."""
    current_version = 0
    if os.path.exists(VERSION_FILE):
        try:
            current_version = int(open(VERSION_FILE).read().strip())
        except (ValueError, OSError):
            pass
    print(f"Current llama.cpp version: b{current_version}")

    resp = requests.get('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest', timeout=15)
    resp.raise_for_status()
    release = resp.json()
    tag = release['tag_name']
    latest_version = int(tag.lstrip('b'))
    print(f"Latest llama.cpp version:  {tag}")

    if current_version >= latest_version:
        print("Already up to date.")
        return

    assets = {a['name']: a for a in release['assets']}
    bin_name = f'llama-{tag}-bin-win-cuda-{CUDA_VERSION}-x64.zip'
    cudart_name = f'cudart-llama-bin-win-cuda-{CUDA_VERSION}-x64.zip'

    if bin_name not in assets:
        print(f"Asset '{bin_name}' not found. Available CUDA assets:")
        for name in sorted(assets):
            if 'cuda' in name.lower():
                print(f"  {name}")
        return

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.name().lower() == 'llama-server.exe':
                print(f"Stopping running llama-server (PID {proc.pid})...")
                proc.kill()
                proc.wait(timeout=10)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    os.makedirs(LLAMA_CPP_DIR, exist_ok=True)

    def download_and_extract(asset_name):
        asset = assets[asset_name]
        size_mb = asset['size'] // (1024 * 1024)
        print(f"Downloading {asset_name} ({size_mb} MB)...")
        dl = requests.get(asset['browser_download_url'], stream=True, timeout=300)
        dl.raise_for_status()
        buf = io.BytesIO()
        for chunk in dl.iter_content(chunk_size=65536):
            buf.write(chunk)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(LLAMA_CPP_DIR)
        print(f"  Extracted {len(zf.namelist())} files")

    download_and_extract(bin_name)
    if cudart_name in assets:
        download_and_extract(cudart_name)

    with open(VERSION_FILE, 'w') as f:
        f.write(str(latest_version))
    print(f"Updated successfully: b{current_version} -> b{latest_version}")


# ============================================
# Server management
# ============================================
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port:
                    print(f"Killing existing process on port {port}: {proc.name()} (PID: {proc.pid})")
                    proc.kill()
                    time.sleep(2)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def download_model_if_missing(model_name):
    """Download model GGUF from Hugging Face if not present locally."""
    config = MODEL_CONFIGS[model_name]
    model_path = config['model_path']
    if os.path.exists(model_path):
        return
    hf_repo = config.get('hf_repo')
    if not hf_repo:
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Add 'hf_repo' to the model config to enable auto-download."
        )
    filename = os.path.basename(model_path)
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
    print(f"Model not found locally. Downloading from Hugging Face...")
    print(f"  Repo : {hf_repo}")
    print(f"  File : {filename}")
    print(f"  Dest : {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024} / {total // 1024 // 1024} MB)", end='', flush=True)
    print(f"\nDownload complete: {model_path}")


def start_server(model_name):
    """Start llama-server with the specified model configuration."""
    global llama_server_process

    if model_name not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    download_model_if_missing(model_name)

    selected_config = MODEL_CONFIGS[model_name]

    print(f"Selected Model: {model_name}")
    print(f"Description: {selected_config['description']}")
    print(f"Path: {selected_config['model_path']}")
    print(f"Context: {selected_config['context_size']} tokens")
    print(f"GPU Layers: {selected_config['gpu_layers']}")
    flash_attn_val = selected_config.get('flash_attn', False)
    print(f"Flash Attention: {'on' if flash_attn_val and str(flash_attn_val).lower() not in ('off', 'false', '0') else 'off'}")
    if selected_config.get('override_tensors'):
        for pattern, device in selected_config['override_tensors']:
            print(f"Override: {pattern} -> {device}")
    print(f"Jinja Template: {'Enabled' if selected_config.get('jinja') else 'Disabled'}")
    if 'reasoning_budget' in selected_config:
        print(f"Reasoning Budget: {selected_config['reasoning_budget']}")
    if 'enable_thinking' in selected_config:
        print(f"Enable Thinking (chat template): {selected_config['enable_thinking']}")

    sampling_params = []
    for p in ['temp', 'top_p', 'top_k', 'min_p', 'repeat_penalty']:
        if p in selected_config:
            sampling_params.append(f"{p}={selected_config[p]}")
    if sampling_params:
        print(f"Sampling: {', '.join(sampling_params)}")

    if llama_server_process:
        print("\nStopping existing server...")
        llama_server_process.terminate()
        llama_server_process.wait()

    if is_port_in_use(8080):
        kill_process_on_port(8080)

    cmd = [
        r'c:\llm\llama\llama-server.exe',
        '-m', selected_config['model_path'],
        '-c', str(selected_config['context_size']),
        '-n', '-1',
        '--port', '8080',
    ]
    if 'reasoning_budget' in selected_config:
        cmd.extend(['--reasoning-budget', str(selected_config['reasoning_budget'])])

    flash_attn = selected_config.get('flash_attn')
    if flash_attn:
        cmd.extend(['--flash-attn', str(flash_attn)])

    if selected_config['gpu_layers'] != 'auto':
        cmd.extend(['-ngl', str(selected_config['gpu_layers'])])

    if selected_config.get('jinja'):
        cmd.append('--jinja')

    if selected_config.get('chat_template_file'):
        cmd.extend(['--chat-template-file', selected_config['chat_template_file']])

    fit = selected_config.get('fit')
    if fit:
        cmd.extend(['--fit', 'on' if fit is True else str(fit)])

    server_env = os.environ.copy()
    if 'enable_thinking' in selected_config and not selected_config.get('chat_template_file'):
        server_env['LLAMA_CHAT_TEMPLATE_KWARGS'] = _json.dumps({"enable_thinking": selected_config['enable_thinking']})

    if 'batch_size' in selected_config:
        cmd.extend(['-b', str(selected_config['batch_size'])])

    if 'parallel' in selected_config:
        cmd.extend(['--parallel', str(selected_config['parallel'])])

    if selected_config.get('override_tensors'):
        for pattern, device in selected_config['override_tensors']:
            cmd.extend(['--override-tensor', f'{pattern}={device}'])

    for param, flag in [('temp', '--temp'), ('top_p', '--top-p'), ('top_k', '--top-k'),
                         ('min_p', '--min-p'), ('repeat_penalty', '--repeat-penalty')]:
        if param in selected_config:
            cmd.extend([flag, str(selected_config[param])])

    print(f"\nStarting llama-server...")
    print(f"   Command: {' '.join(cmd)}")
    if 'LLAMA_CHAT_TEMPLATE_KWARGS' in server_env:
        print(f"   Env: LLAMA_CHAT_TEMPLATE_KWARGS={server_env['LLAMA_CHAT_TEMPLATE_KWARGS']}")

    log_path = os.path.join(output_dir, 'llama_server.log')
    print(f"   Log: {log_path}")
    _log_file = open(log_path, 'w', encoding='utf-8')
    llama_server_process = subprocess.Popen(
        cmd,
        stdout=_log_file,
        stderr=_log_file,
        env=server_env
    )

    print("Waiting for server to start...")
    for i in range(120):
        try:
            resp = requests.get("http://127.0.0.1:8080/health", timeout=2)
            if resp.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(".", end="", flush=True)
        time.sleep(1)

    print("\nServer failed to start within 120 seconds!")
    return False


def stop_server():
    """Stop the llama-server."""
    global llama_server_process
    if llama_server_process:
        llama_server_process.terminate()
        llama_server_process.wait()
        llama_server_process = None
        print("Server stopped.")
    else:
        if kill_process_on_port(8080):
            print("Killed server on port 8080.")
        else:
            print("No server running.")


# ============================================
# Semantic model loading
# ============================================
def _yaml_to_system_prompt(parsed: dict) -> str:
    """Convert an OSI v0.1.1 YAML semantic model to a plain-text system prompt."""
    models = parsed.get('semantic_model', [])
    if not models:
        return ""
    m = models[0]

    lines = []

    desc = m.get('description', '')
    if desc:
        lines.append(f"# {desc}")

    ai_ctx = m.get('ai_context', {})
    instructions = ai_ctx.get('instructions', '') if isinstance(ai_ctx, dict) else str(ai_ctx)
    if instructions:
        lines.append("")
        for instruction_line in instructions.rstrip().splitlines():
            lines.append(instruction_line)

    # Schema section
    datasets = m.get('datasets', [])
    if datasets:
        lines.append("")
        lines.append("# ============================================================")
        lines.append("# SCHEMA")
        lines.append("# ============================================================")
        for ds in datasets:
            ds_name = ds.get('name', '')
            ds_desc = ds.get('description', '')
            pk = ds.get('primary_key', [])
            pk_str = f"  (PK: {', '.join(pk)})" if pk else ""
            lines.append(f"")
            lines.append(f"# --- {ds_name}{pk_str} ---")
            if ds_desc:
                lines.append(f"#   {ds_desc}")
            for field in ds.get('fields', []):
                f_name = field.get('name', '')
                f_desc = field.get('description', '')
                f_desc_str = f"  -- {f_desc}" if f_desc else ""
                lines.append(f"#   {f_name}{f_desc_str}")

    # Relationships section
    relationships = m.get('relationships', [])
    if relationships:
        lines.append("")
        lines.append("# ============================================================")
        lines.append("# RELATIONSHIPS")
        lines.append("# ============================================================")
        for rel in relationships:
            from_ds = rel.get('from', '')
            to_ds = rel.get('to', '')
            from_cols = rel.get('from_columns', [])
            to_cols = rel.get('to_columns', [])
            lines.append(f"#   {from_ds}.{from_cols} → {to_ds}.{to_cols}")

    # Metrics section
    metrics = m.get('metrics', [])
    if metrics:
        lines.append("")
        lines.append("# ============================================================")
        lines.append("# METRICS")
        lines.append("# ============================================================")
        for metric in metrics:
            m_name = metric.get('name', '')
            m_desc = metric.get('description', '')
            dialects = metric.get('expression', {}).get('dialects', [])
            expr = dialects[0].get('expression', '') if dialects else ''
            lines.append(f"#   {m_name} = {expr}")
            if m_desc:
                lines.append(f"#     {m_desc}")

    return "\n".join(lines)


def load_semantic_model(url_or_path: str) -> str:
    """Load semantic model from a URL or local file path.

    Supports plain .txt files and OSI v0.1.1 .yaml files.
    YAML files are converted to a system prompt via _yaml_to_system_prompt().
    """
    raw = ""
    try:
        resp = requests.get(url_or_path)
        resp.raise_for_status()
        raw = resp.text.strip()
    except requests.RequestException:
        try:
            with open(url_or_path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
        except Exception as e:
            return f"Error loading system prompt: {e}"

    if url_or_path.endswith('.yaml') or url_or_path.endswith('.yml') or raw.startswith('version:'):
        try:
            parsed = yaml.safe_load(raw)
            return _yaml_to_system_prompt(parsed)
        except Exception as e:
            return f"Error parsing YAML semantic model: {e}"

    return raw


# ============================================
# LLM interaction
# ============================================
def get_ai_response(user_message, model_name, endpoint=None):
    """Generate SQL query using llama.cpp server."""
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    user_message = str(user_message)

    system_prompt = load_semantic_model(test_semantic_model_url)
    if system_prompt.startswith("Error"):
        return system_prompt

    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'stream': False
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        message = data.get('choices', [{}])[0].get('message', {})
        generated_text = message.get('content', '') or message.get('reasoning_content', '')

        timings = data.get('timings', {})
        usage = data.get('usage', {})
        _token_accumulator["completion_tokens"] += usage.get('completion_tokens', 0) or timings.get('predicted_n', 0)
        _token_accumulator["generation_ms"] += timings.get('predicted_ms', 0.0)

        if not generated_text:
            return f"No content received from llama.cpp server. Response: {data}"

    except requests.RequestException as e:
        return f"Error connecting to llama.cpp server at {endpoint}: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

    if not isinstance(generated_text, str):
        return f"Invalid response type: {type(generated_text)}"

    cleaned_text = generated_text
    cleaned_text = re.sub(r'<\|[^|]*\|>', '', cleaned_text).strip()
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned_text = re.sub(r'.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()

    sql_in_blocks = re.findall(r'```(?:sql|duckdb)?\s*([\s\S]*?)\s*```', cleaned_text, flags=re.IGNORECASE)
    if sql_in_blocks:
        cleaned_text = sql_in_blocks[-1].strip()

    cleaned_text = cleaned_text.replace("```sql", "").replace("```duckdb", "").replace("```", "").strip()

    double_dash_pos = cleaned_text.find('--')
    if double_dash_pos != -1:
        comment_end = cleaned_text.find('\n', double_dash_pos)
        if comment_end != -1:
            sql_after_comment = cleaned_text[comment_end + 1:].strip()
            if sql_after_comment:
                return sql_after_comment

    return cleaned_text


def get_feedback_response(original_question, sql_query, query_results, model_name, sql_error=None, conversation_history=None, endpoint=None):
    """Get LLM feedback on a SQL query.

    If sql_error is provided: the SQL failed — ask the model to fix it.
    Otherwise: the SQL ran — show results and ask if they correctly answer the question.
    """
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    original_question = str(original_question)

    if sql_error:
        feedback_prompt = f"""QUESTION: {original_question}

SQL:
{sql_query}

ERROR:
{sql_error}

The SQL failed. Return only a corrected SQL query (no explanation, no markdown)."""
    else:
        if isinstance(query_results, pd.DataFrame):
            if len(query_results) > 10:
                results_preview = f"Results (showing first 10 of {len(query_results)} rows):\n{query_results.head(10).to_string()}\n\nTotal rows: {len(query_results)}"
            else:
                results_preview = f"Results ({len(query_results)} rows):\n{query_results.to_string()}"

            quality_analysis = []

            if len(query_results.columns) == 1:
                col_name = query_results.columns[0]
                total_rows = len(query_results)
                unique_rows = query_results[col_name].nunique()
                if total_rows > unique_rows:
                    duplicate_count = total_rows - unique_rows
                    quality_analysis.append(f"POTENTIAL ISSUE: Column '{col_name}' has {duplicate_count} duplicates. For 'list categories' questions, duplicates are usually wrong.")

            if "categor" in original_question.lower() or "list" in original_question.lower():
                if len(query_results.columns) == 1:
                    col_name = query_results.columns[0]
                    if query_results[col_name].nunique() < len(query_results):
                        quality_analysis.append(f"POTENTIAL ISSUE: 'List categories' question has duplicate results. Consider using DISTINCT.")

            if "order" in original_question.lower() or "sort" in original_question.lower():
                if "alphabetical" in original_question.lower():
                    quality_analysis.append(f"REMINDER: Question asks for alphabetical ordering. Verify ORDER BY clause.")

            quality_section = "\n\nQUALITY ANALYSIS:\n" + "\n".join(quality_analysis) if quality_analysis else ""
        else:
            results_preview = "No results returned (empty or error)"
            quality_section = ""

        feedback_prompt = f"""QUESTION: {original_question}

SQL:
{sql_query}

RESULTS:
{results_preview}{quality_section}

If the results correctly answer the question, respond with: RESULTS_CORRECT
Otherwise, return only a revised SQL query (no explanation, no markdown)."""

    system_prompt = load_semantic_model(test_semantic_model_url)
    if system_prompt.startswith("Error"):
        return system_prompt

    messages = [{'role': 'system', 'content': system_prompt}]
    if conversation_history:
        messages += conversation_history
    messages.append({'role': 'user', 'content': feedback_prompt})

    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'messages': messages,
            'stream': False
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        message = data.get('choices', [{}])[0].get('message', {})
        generated_text = message.get('content', '') or message.get('reasoning_content', '')

        timings = data.get('timings', {})
        usage = data.get('usage', {})
        _token_accumulator["completion_tokens"] += usage.get('completion_tokens', 0) or timings.get('predicted_n', 0)
        _token_accumulator["generation_ms"] += timings.get('predicted_ms', 0.0)

        if not generated_text:
            return f"No feedback received from llama.cpp server"

    except Exception as e:
        return f"Error in feedback loop: {e}"

    if not isinstance(generated_text, str):
        return "Invalid feedback response type"

    cleaned_text = generated_text
    cleaned_text = re.sub(r'<\|[^|]*\|>', '', cleaned_text).strip()
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned_text = re.sub(r'.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()

    sql_in_blocks = re.findall(r'```(?:sql|duckdb)?\s*([\s\S]*?)\s*```', cleaned_text, flags=re.IGNORECASE)
    if sql_in_blocks:
        cleaned_text = sql_in_blocks[-1].strip()

    cleaned_text = cleaned_text.replace("```sql", "").replace("```duckdb", "").replace("```", "").strip()

    return cleaned_text


# ============================================
# SQL execution
# ============================================
def execute_sql_once(query):
    """Execute SQL once. Returns (DataFrame, None) on success or (None, error_string) on failure."""
    result_container = {"result": None, "error": None}
    current_query = query.strip()

    def query_thread():
        try:
            result_container["result"] = con.execute(current_query).fetchdf()
        except duckdb.InterruptException:
            result_container["error"] = f"Query interrupted after timeout of {TIMEOUT_SECONDS} seconds."
        except Exception as e:
            result_container["error"] = str(e)

    thread = threading.Thread(target=query_thread)
    thread.start()
    start_time = time.time()
    while thread.is_alive():
        if time.time() - start_time > TIMEOUT_SECONDS:
            con.interrupt()
            thread.join()
            return None, f"Query execution timed out after {TIMEOUT_SECONDS} seconds."
        time.sleep(0.01)

    if result_container["error"]:
        return None, result_container["error"]
    return result_container["result"], None


def execute_sql_with_retry(query, test_model):
    """Execute SQL with retry on syntax errors. Returns (result, attempt, query, retry_llm_elapsed)."""
    attempt = 1
    current_query = query.strip()
    retry_llm_elapsed = 0.0

    while attempt <= max_attempts:
        result_container = {"result": None, "error": None}

        def query_thread():
            try:
                result_container["result"] = con.execute(current_query).fetchdf()
            except duckdb.InterruptException:
                result_container["error"] = f"Query interrupted after timeout of {TIMEOUT_SECONDS} seconds."
            except Exception as e:
                result_container["error"] = str(e)

        thread = threading.Thread(target=query_thread)
        thread.start()

        start_time = time.time()
        while thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                con.interrupt()
                thread.join()
                return f"Query execution timed out after {TIMEOUT_SECONDS} seconds.", attempt, "query runs forever", retry_llm_elapsed
            time.sleep(0.01)

        if result_container["error"]:
            print(current_query)
            print(f"Attempt {attempt}/{max_attempts} failed with error: {result_container['error']}")

            if attempt == max_attempts:
                return f"Max attempts reached. Last error: {result_container['error']}", attempt, current_query, retry_llm_elapsed

            message = (
                f"The following SQL query failed: '{current_query}'.\n"
                f"Error message: {result_container['error']}\n"
                f"Please provide the corrected SQL query. Return only the corrected query without explanation."
            )

            t0 = time.time()
            corrected_query = get_ai_response(message, test_model)
            retry_llm_elapsed += time.time() - t0

            if corrected_query.startswith("Error"):
                return f"Failed to get corrected query: {corrected_query}", attempt, current_query, retry_llm_elapsed

            current_query = corrected_query.strip()
            attempt += 1
        else:
            return result_container["result"], attempt, current_query, retry_llm_elapsed

    return "Unexpected error or loop termination", attempt, current_query, retry_llm_elapsed


# ============================================
# Test runner
# ============================================
def ask_question(questions_list, test_model):
    """Process questions and generate SQL queries using llama.cpp server."""
    results_data = []
    for i, x in enumerate(questions_list):
        print(f"Question {i+1}: {x}")
        _token_accumulator["completion_tokens"] = 0
        _token_accumulator["generation_ms"] = 0.0
        start_time = time.time()
        llm_elapsed = 0.0
        try:
            t0 = time.time(); sql_query_or_error = get_ai_response(x, test_model); llm_elapsed += time.time() - t0
            query_result_data_json = []
            attempts_count = None
            error_details = None
            feedback_iterations = 0
            final_sql_query = sql_query_or_error

            if sql_query_or_error is None or sql_query_or_error.startswith("Error"):
                error_message = sql_query_or_error if sql_query_or_error is not None else "AI response was None"
                error_details = f"AI Error: {error_message}"
                result_row_count = 0
            else:
                current_query = sql_query_or_error
                current_result = None
                print(f"SQL: {current_query}")

                # Phase 1: fix broken SQL (up to max_attempts retries on error)
                for attempt in range(1, max_attempts + 1):
                    result_df, sql_error = execute_sql_once(current_query)
                    if not sql_error:
                        current_result = result_df
                        break
                    print(f"Attempt {attempt} error: {sql_error}")
                    if not enable_feedback_loop or attempt >= max_attempts:
                        error_details = f"Execution Error: {sql_error}"
                        break
                    t0 = time.time()
                    fix_response = get_feedback_response(x, current_query, None, test_model, sql_error=sql_error)
                    llm_elapsed += time.time() - t0
                    feedback_iterations += 1
                    if fix_response.startswith("Error"):
                        error_details = f"Execution Error: {sql_error}"
                        break
                    current_query = fix_response.strip()
                    final_sql_query = current_query

                attempts_count = attempt

                # Phase 2: confirm results once — if model revises, run new SQL and accept it
                if current_result is not None and enable_feedback_loop:
                    t0 = time.time()
                    confirm_response = get_feedback_response(x, current_query, current_result, test_model)
                    llm_elapsed += time.time() - t0
                    feedback_iterations += 1
                    if "RESULTS_CORRECT" not in confirm_response.upper() and not confirm_response.startswith("Error"):
                        revised_df, revised_error = execute_sql_once(confirm_response.strip())
                        if not revised_error:
                            current_query = confirm_response.strip()
                            current_result = revised_df
                            final_sql_query = current_query
                        else:
                            print(f"Revised SQL failed ({revised_error}), keeping original result.")

                if current_result is not None:
                    display(current_result)
                    query_result_data_json = current_result.to_dict('records')
                    error_details = None
                    result_row_count = len(current_result)
                else:
                    if not error_details:
                        error_details = f"Execution Error: query failed after {attempts_count} attempts"
                    print("Execution: FAILED")
                    result_row_count = 0

            end_time = time.time()
            duration = round(end_time - start_time, 2)
            exec_elapsed = duration - llm_elapsed
            completion_tokens = _token_accumulator["completion_tokens"]
            generation_s = round(_token_accumulator["generation_ms"] / 1000, 2)
            tokens_per_s = round(completion_tokens / generation_s, 1) if generation_s > 0 else None
            print(f" ### LLM: {llm_elapsed:.1f}s | Exec: {exec_elapsed:.1f}s | Total: {duration:.1f}s ###")
            results_data.append({
                "model": test_model,
                "SF": SF,
                "semantic_model": os.path.splitext(os.path.basename(test_semantic_model_url))[0],
                "timestamp": timestamp,
                "nbr": i + 1,
                "question": x,
                "duration_s": duration,
                "completion_tokens": completion_tokens,
                "generation_s": generation_s,
                "tokens_per_s": tokens_per_s,
                "sql_query": final_sql_query,
                "attempts": attempts_count,
                "feedback_iterations": feedback_iterations,
                "result": query_result_data_json,
                "result_count": result_row_count,
                "error_details": error_details
            })
        except Exception as _e:
            import traceback
            print(f"CRASH on Q{i+1}: {_e}")
            traceback.print_exc()
            raise

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    sanitized_model = re.sub(r'[\\/*?:"<>|]', '_', test_model)
    output_filename = f"{timestamp}_{sanitized_model}.json"
    output_path = os.path.join(log_dir, output_filename)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        return f"Successfully processed {len(questions_list)} questions. Results saved to {output_path}"
    except IOError as e:
        return f"Error saving results to {output_path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred during file saving: {e}"


# ============================================
# Result comparison & analysis
# ============================================
def compare_query_results(ref_sql, model_sql, connection, rtol=0.005):
    """Compare two SQL query results by re-executing them and matching columns by value similarity."""
    try:
        ref_df = connection.execute(ref_sql).fetchdf()
    except Exception:
        return 'error'
    try:
        model_df = connection.execute(model_sql).fetchdf()
    except Exception:
        return 'error'

    if len(ref_df) == 0 and len(model_df) == 0:
        return 'exact'

    if len(ref_df) != len(model_df):
        return 'row_mismatch'

    n_rows = len(ref_df)
    n_ref_cols = len(ref_df.columns)
    n_mod_cols = len(model_df.columns)

    ref_sorted = ref_df.sort_values(by=list(ref_df.columns)).reset_index(drop=True)
    model_sorted = model_df.sort_values(by=list(model_df.columns)).reset_index(drop=True)

    def columns_match(ref_col, mod_col, rtol):
        matches = 0
        for rv, mv in zip(ref_col, mod_col):
            if pd.isna(rv) and pd.isna(mv):
                matches += 1
                continue
            if pd.isna(rv) or pd.isna(mv):
                continue
            try:
                rv_f = float(rv)
                mv_f = float(mv)
                denom = max(abs(rv_f), abs(mv_f))
                if denom == 0:
                    if rv_f == mv_f:
                        matches += 1
                elif abs(rv_f - mv_f) / denom <= rtol:
                    matches += 1
                continue
            except (ValueError, TypeError):
                pass
            if str(rv).strip() == str(mv).strip():
                matches += 1
        return matches / max(n_rows, 1)

    used_mod_cols = set()
    matched_ref_cols = 0

    for ref_col_name in ref_sorted.columns:
        best_score = 0
        best_mod_col = None
        for mod_col_name in model_sorted.columns:
            if mod_col_name in used_mod_cols:
                continue
            score = columns_match(ref_sorted[ref_col_name], model_sorted[mod_col_name], rtol)
            if score > best_score:
                best_score = score
                best_mod_col = mod_col_name
        if best_score == 1.0 and best_mod_col is not None:
            used_mod_cols.add(best_mod_col)
            matched_ref_cols += 1

    if matched_ref_cols == n_ref_cols:
        if n_mod_cols > n_ref_cols:
            return 'superset'
        return 'exact'

    ref_str_cols = [c for c in ref_sorted.columns if ref_sorted[c].dtype == 'object']
    if ref_str_cols:
        str_col_mapping = {}
        for rc in ref_str_cols:
            ref_vals = set(ref_sorted[rc].dropna().astype(str))
            for mc in model_sorted.columns:
                if mc in str_col_mapping.values():
                    continue
                mod_vals = set(model_sorted[mc].dropna().astype(str))
                if len(ref_vals & mod_vals) >= len(ref_vals) * 0.8:
                    str_col_mapping[rc] = mc
                    break

        if str_col_mapping:
            ref_sort_cols = list(str_col_mapping.keys())
            mod_sort_cols = [str_col_mapping[c] for c in ref_sort_cols]

            try:
                ref_sorted2 = ref_df.sort_values(by=ref_sort_cols).reset_index(drop=True)
                model_sorted2 = model_df.sort_values(by=mod_sort_cols).reset_index(drop=True)

                used_mod_cols2 = set()
                matched_ref_cols2 = 0
                for ref_col_name in ref_sorted2.columns:
                    best_score = 0
                    best_mod_col = None
                    for mod_col_name in model_sorted2.columns:
                        if mod_col_name in used_mod_cols2:
                            continue
                        score = columns_match(ref_sorted2[ref_col_name], model_sorted2[mod_col_name], rtol)
                        if score > best_score:
                            best_score = score
                            best_mod_col = mod_col_name
                    if best_score >= 0.95 and best_mod_col is not None:
                        used_mod_cols2.add(best_mod_col)
                        matched_ref_cols2 += 1

                if matched_ref_cols2 == n_ref_cols:
                    if n_mod_cols > n_ref_cols:
                        return 'superset'
                    return 'exact'
            except Exception:
                pass

    return 'mismatch'


def display_side_by_side(nbr, model1, model2):
    """Display results from two models side by side for a given question."""
    print(f"question {nbr} : " + duckdb.sql(f" select question from results_filtered where nbr = {nbr}  ").fetchone()[0])
    try:
        sql_query1 = duckdb.sql(f""" select sql_query from results_filtered where nbr = {nbr} and model = '{model1}' """).fetchone()[0]
        sql_query2 = duckdb.sql(f""" select sql_query from results_filtered where nbr = {nbr} and model = '{model2}' """).fetchone()[0]
        df1 = con.sql(sql_query1).df()
        df2 = con.sql(sql_query2).df()
        side_by_side = pd.concat([df1, df2], axis=1, keys=[model1, model2])
        display(side_by_side)
        print(f"{model1} SQL Query:")
        print("--------------------")
        print(sql_query1)
        print("\n")
        print(f"{model2} SQL Query:")
        print("--------------------")
        print(sql_query2)
        print("\n")
    except Exception as e:
        print(f"Error executing query for nbr {nbr}: {e}")


def analyze_results():
    """Load and analyze test results, validate each model's answers against baseline."""
    log_path = os.path.join(output_dir, 'log', '*.json')
    baseline = model0

    all_models_query = duckdb.sql(f"""
        SELECT DISTINCT model
        FROM read_json_auto('{log_path}', union_by_name=true, ignore_errors=true)
        WHERE SF = '{SF}'
        AND timestamp = (SELECT MAX(timestamp) FROM read_json_auto('{log_path}', union_by_name=true, ignore_errors=true) r2 WHERE r2.model = read_json_auto.model AND r2.SF = '{SF}')
        ORDER BY model
    """).df()

    available_models = all_models_query['model'].tolist()
    models_sql_filter = "', '".join(available_models)

    duckdb.sql(f"""
        CREATE OR REPLACE VIEW results_all AS
        SELECT *
        FROM read_json_auto('{log_path}', union_by_name=true, ignore_errors=true)
        WHERE model in ('{models_sql_filter}') and SF = '{SF}'
    """)
    duckdb.sql(f"""
        CREATE OR REPLACE TEMP TABLE results_filtered AS
        SELECT * FROM results_all
        WHERE timestamp = (SELECT max(timestamp) FROM results_all r2 WHERE r2.model = results_all.model)
    """)
    try:
        duckdb.sql("SELECT feedback_iterations FROM results_filtered LIMIT 1")
    except Exception:
        duckdb.sql("ALTER TABLE results_filtered ADD COLUMN feedback_iterations INTEGER DEFAULT 0")

    eval_models = [m for m in available_models if m != baseline]
    print(f"Evaluating {len(eval_models)} models: {', '.join(eval_models)}")

    ref_queries = duckdb.sql(f"""
        SELECT nbr, sql_query, error_details
        FROM results_filtered WHERE model = '{baseline}'
        ORDER BY nbr
    """).df()
    ref_map = {row['nbr']: row['sql_query'] for _, row in ref_queries.iterrows()
               if row['error_details'] is None or str(row['error_details']) == 'None'}

    q_labels = duckdb.sql(f"""
        SELECT DISTINCT nbr, question FROM results_filtered ORDER BY nbr
    """).df()
    q_map = dict(zip(q_labels['nbr'], q_labels['question']))

    all_details = []
    all_summaries = []

    for comp_model in eval_models:
        model_queries = duckdb.sql(f"""
            SELECT nbr, sql_query, error_details, duration_s,
                   COALESCE(feedback_iterations, 0) as feedback_iterations
            FROM results_filtered WHERE model = '{comp_model}'
            ORDER BY nbr
        """).df()

        scores = []
        for _, mrow in model_queries.iterrows():
            nbr = int(mrow['nbr'])
            has_error = mrow['error_details'] is not None and str(mrow['error_details']) != 'None'

            if has_error:
                verdict = 'error'
            elif nbr not in ref_map:
                verdict = 'no_baseline'
            else:
                verdict = compare_query_results(ref_map[nbr], mrow['sql_query'], con)

            is_correct = verdict == 'exact'
            is_raw_correct = is_correct and int(mrow['feedback_iterations']) == 0
            scores.append({
                'model': comp_model, 'nbr': nbr, 'verdict': verdict,
                'correct': is_correct,
                'raw_correct': is_raw_correct,
                'duration_s': mrow['duration_s'],
                'feedback_iterations': mrow['feedback_iterations']
            })
            all_details.append({
                'model': comp_model, 'nbr': nbr,
                'question': q_map.get(nbr, '')[:60],
                'verdict': verdict,
                'correct': 'Y' if is_correct else '',
                'duration_s': mrow['duration_s']
            })

        scores_df = pd.DataFrame(scores)
        n_correct = int(scores_df['correct'].sum())
        n_raw_correct = int(scores_df['raw_correct'].sum())
        n_total = len(scores_df)

        all_summaries.append({
            'model': comp_model,
            'correct': n_correct,
            'incorrect': n_total - n_correct,
            'accuracy_percent': round(n_correct / max(n_total, 1) * 100, 1),
            'raw_accuracy_percent': round(n_raw_correct / max(n_total, 1) * 100, 1),
            'avg_duration_per_question': round(scores_df['duration_s'].mean(), 2),
            'total_feedback_iterations': int(scores_df['feedback_iterations'].sum()),
        })

    if all_details:
        detail_df = pd.DataFrame(all_details)
        pivot = detail_df.pivot(index='nbr', columns='model', values='verdict')
        pivot.columns.name = None
        pivot.index.name = 'Q#'
        pivot.insert(0, 'question', pivot.index.map(lambda n: q_map.get(n, '')[:55]))
        print("\nPer-question validation:")
        display(pivot)

    if all_summaries:
        comparison_table = pd.DataFrame(all_summaries)
        comparison_table = comparison_table.sort_values('accuracy_percent', ascending=False).reset_index(drop=True)
    else:
        comparison_table = pd.DataFrame(columns=[
            'model', 'correct', 'incorrect', 'accuracy_percent', 'raw_accuracy_percent',
            'avg_duration_per_question', 'total_feedback_iterations'
        ])

    print("\nSummary:")
    display(comparison_table)
    return comparison_table


def _compare_stored_results(ref_result, model_result, rtol=0.005):
    """Compare two stored result lists (from JSON logs) without re-executing SQL."""
    try:
        ref_df = pd.DataFrame(ref_result) if ref_result else pd.DataFrame()
        mod_df = pd.DataFrame(model_result) if model_result else pd.DataFrame()
    except Exception:
        return 'error'

    if len(ref_df) == 0 and len(mod_df) == 0:
        return 'exact'

    if len(ref_df) != len(mod_df):
        return 'row_mismatch'

    n_rows = len(ref_df)
    n_ref_cols = len(ref_df.columns)

    ref_sorted = ref_df.sort_values(by=list(ref_df.columns)).reset_index(drop=True)
    mod_sorted = mod_df.sort_values(by=list(mod_df.columns)).reset_index(drop=True)

    def columns_match(ref_col, mod_col):
        matches = 0
        for rv, mv in zip(ref_col, mod_col):
            if pd.isna(rv) and pd.isna(mv):
                matches += 1
                continue
            if pd.isna(rv) or pd.isna(mv):
                continue
            try:
                rv_f, mv_f = float(rv), float(mv)
                denom = max(abs(rv_f), abs(mv_f))
                if denom == 0:
                    if rv_f == mv_f:
                        matches += 1
                elif abs(rv_f - mv_f) / denom <= rtol:
                    matches += 1
                continue
            except (ValueError, TypeError):
                pass
            if str(rv).strip() == str(mv).strip():
                matches += 1
        return matches / max(n_rows, 1)

    def _try_match(rs, ms):
        used = set()
        found = 0
        for rc in rs.columns:
            for mc in ms.columns:
                if mc in used:
                    continue
                if columns_match(rs[rc], ms[mc]) == 1.0:
                    used.add(mc)
                    found += 1
                    break
        return found

    matched = _try_match(ref_sorted, mod_sorted)
    if matched == n_ref_cols:
        return 'exact'

    # Fallback: re-sort by matched string columns to handle column-order swaps
    ref_str_cols = [c for c in ref_df.columns if ref_df[c].dtype == 'object']
    if ref_str_cols:
        str_col_mapping = {}
        for rc in ref_str_cols:
            ref_vals = set(ref_df[rc].dropna().astype(str))
            for mc in mod_df.columns:
                if mc in str_col_mapping.values():
                    continue
                mod_vals = set(mod_df[mc].dropna().astype(str))
                if len(ref_vals & mod_vals) >= len(ref_vals) * 0.8:
                    str_col_mapping[rc] = mc
                    break
        if str_col_mapping:
            ref_sort_cols = list(str_col_mapping.keys())
            mod_sort_cols = [str_col_mapping[c] for c in ref_sort_cols]
            try:
                ref_sorted2 = ref_df.sort_values(by=ref_sort_cols).reset_index(drop=True)
                mod_sorted2 = mod_df.sort_values(by=mod_sort_cols).reset_index(drop=True)
                if _try_match(ref_sorted2, mod_sorted2) == n_ref_cols:
                    return 'exact'
            except Exception:
                pass

    return 'wrong'


def analyze_all_runs(date=None):
    """Load and score ALL historical runs using stored results (no SQL re-execution)."""
    import glob as _glob
    from collections import defaultdict

    baseline = model0

    # Load all log files directly — stored results are already there, no need to re-run SQL
    all_records = []
    for f in _glob.glob(os.path.join(output_dir, 'log', '*.json')):
        with open(f, 'r', encoding='utf-8') as fh:
            all_records.extend(json.load(fh))
    all_records = [r for r in all_records if str(r.get('SF', '')) == str(SF)]

    # Build baseline reference map (latest run) — always from full history
    baseline_records = [r for r in all_records if r.get('model') == baseline]
    latest_ts = max(r['timestamp'] for r in baseline_records)
    ref_map = {
        r['nbr']: r.get('result', [])
        for r in baseline_records
        if r['timestamp'] == latest_ts
        and not r.get('error_details')
    }

    # Group model runs by (model, timestamp)
    date_prefix = date.replace('-', '') if date is not None else None
    runs = defaultdict(list)
    for r in all_records:
        if r.get('model') != baseline:
            if date_prefix is None or str(r.get('timestamp', '')) >= date_prefix:
                runs[(r['model'], r['timestamp'])].append(r)

    all_summaries = []
    for (run_model, run_ts), records in sorted(runs.items()):
        scores = []
        for r in records:
            if r.get('error_details'):
                correct = False
            elif r['nbr'] not in ref_map:
                correct = False
            else:
                correct = _compare_stored_results(ref_map[r['nbr']], r.get('result', [])) == 'exact'
            tps = r.get('tokens_per_s')
            scores.append({'correct': correct, 'duration_s': r.get('duration_s', 0), 'tokens_per_s': tps, 'completion_tokens': r.get('completion_tokens', 0)})

        if not scores:
            continue
        n_correct = sum(s['correct'] for s in scores)
        n_total = len(scores)
        tps_values = [s['tokens_per_s'] for s in scores if s['tokens_per_s']]
        sem_model = records[0].get('semantic_model', 'semantic_model') or 'semantic_model'
        sem_model = os.path.splitext(sem_model)[0]
        all_summaries.append({
            'model': run_model,
            'timestamp': run_ts,
            'semantic_model': sem_model,
            'accuracy_percent': round(n_correct / max(n_total, 1) * 100, 1),
            'avg_duration_per_question': round(sum(s['duration_s'] for s in scores) / max(n_total, 1), 2),
            'avg_tokens_per_s': round(sum(tps_values) / len(tps_values), 1) if tps_values else None,
            'avg_completion_tokens': round(sum(s['completion_tokens'] for s in scores) / max(n_total, 1), 1),
        })

    return pd.DataFrame(all_summaries)


def plot_results(comparison_table, models=None, semantic_model=None):
    """Plot scatter chart of Speed vs Accuracy and save to chart.png.

    Accepts either:
    - a single-row-per-model table (columns: model, accuracy_percent, avg_duration_per_question)
    - an all-runs table from analyze_all_runs() (adds a 'timestamp' column)

    Args:
        models: optional list of model names to include
        semantic_model: optional list of semantic model names to include
    """
    import matplotlib.pyplot as plt

    if models is not None:
        comparison_table = comparison_table[comparison_table['model'].isin(models)]

    if semantic_model is not None:
        sem_col_check = 'semantic_model' if 'semantic_model' in comparison_table.columns else None
        if sem_col_check:
            import pathlib
            normalized = [pathlib.Path(s).stem if pathlib.Path(s).suffix else s for s in semantic_model]
            comparison_table = comparison_table[comparison_table[sem_col_check].isin(normalized)]

    if comparison_table.empty:
        print("plot_results: no data after filtering — check models/semantic_model args")
        return

    base_colors = ['#1f77b4', '#2e8b57', '#ff8c00', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    has_all_runs = 'timestamp' in comparison_table.columns

    sem_col = 'semantic_model' if 'semantic_model' in comparison_table.columns else None
    unique_sem_models = sorted(comparison_table[sem_col].fillna('semantic_model').unique()) if sem_col else ['semantic_model']
    sem_markers = {s: m for s, m in zip(unique_sem_models, ['o', 'D', 's', '^', 'v'])}

    plt.figure(figsize=(12, 7))

    if has_all_runs:
        unique_models = sorted(comparison_table['model'].unique())
        model_color = {m: base_colors[i % len(base_colors)] for i, m in enumerate(unique_models)}

        for model in unique_models:
            mdf = comparison_table[comparison_table['model'] == model].sort_values('timestamp')
            for sem, sdf in mdf.groupby(sem_col if sem_col else lambda x: 'semantic_model'):
                sem = sem or 'semantic_model'
                plt.scatter(sdf['avg_duration_per_question'], sdf['accuracy_percent'],
                            c=model_color[model], marker=sem_markers.get(sem, 'o'),
                            s=40, alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)
            plt.scatter([], [], c=model_color[model], s=40, label=model)

        # Legend: models (color)
        model_legend = plt.legend(loc='lower right', fontsize=10, title='Model')
        plt.gca().add_artist(model_legend)
        # Legend: semantic model (shape)
        import matplotlib.lines as mlines
        sem_handles = [mlines.Line2D([], [], color='grey', marker=sem_markers[s], linestyle='None', markersize=6, label=s) for s in unique_sem_models]
        plt.legend(handles=sem_handles, loc='upper right', fontsize=9, title='Semantic model')

        n_label = f'{len(unique_models)} models, {len(comparison_table)} runs'
        all_times = comparison_table['avg_duration_per_question'].tolist()
    else:
        models = comparison_table['model'].tolist()
        accuracy = comparison_table['accuracy_percent'].tolist()
        avg_time = comparison_table['avg_duration_per_question'].tolist()
        colors = (base_colors * ((len(models) // len(base_colors)) + 1))[:len(models)]
        sem_list = comparison_table[sem_col].fillna('semantic_model').tolist() if sem_col else ['semantic_model'] * len(models)
        for i, (x, y, c, s) in enumerate(zip(avg_time, accuracy, colors, sem_list)):
            plt.scatter(x, y, c=c, marker=sem_markers.get(s, 'o'), s=40, alpha=0.7, edgecolors='black', linewidth=2)
            plt.annotate(models[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
        n_label = f'{len(models)} models'
        all_times = avg_time

    plt.xlabel('Avg Duration (seconds) - lower is better', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    llama_version = 'unknown'
    try:
        llama_version = 'b' + open(r'c:\llm\llama\version.txt').read().strip()
    except Exception:
        pass
    plt.title(f'Text to SQL: Duration vs Accuracy ({n_label})\n(NVIDIA RTX 2000 Mobile, 4GB VRAM, 32GB RAM, llama.cpp {llama_version})', fontsize=14, fontweight='bold')
    import numpy as np
    xmax = max(np.ceil(max(all_times) / 5) * 5, 5)
    plt.xlim(0, xmax)
    all_acc = comparison_table['accuracy_percent'].tolist()
    plt.ylim(0, 100)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, xmax + 5, 5))
    ax.set_yticks(np.arange(0, 110, 10))
    plt.grid(True, alpha=0.3)
    ax.add_patch(plt.matplotlib.patches.Rectangle(
        (0, 90), 5, 10, transform=ax.transData,
        facecolor='#00cc44', alpha=0.12, zorder=0
    ))

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {chart_path}')
    plt.show()


def plot_model_history(all_runs, model_name):
    """Plot all runs of a single model over time: accuracy vs duration, labeled by date.

    Args:
        all_runs: DataFrame from analyze_all_runs()
        model_name: exact model name string (must match 'model' column)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mdf = all_runs[all_runs['model'] == model_name].copy()
    if mdf.empty:
        print(f"No runs found for model '{model_name}'")
        return

    mdf = mdf.sort_values('timestamp')
    mdf['run_label'] = mdf['timestamp'].str[:10]  # YYYY-MM-DD

    sem_col = 'semantic_model' if 'semantic_model' in mdf.columns else None
    unique_sem = sorted(mdf[sem_col].fillna('semantic_model').unique()) if sem_col else ['semantic_model']
    sem_markers = {s: m for s, m in zip(unique_sem, ['o', 'D', 's', '^', 'v'])}

    cmap = plt.cm.plasma
    n = len(mdf)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (_, row) in enumerate(mdf.iterrows()):
        sem = (row[sem_col] if sem_col else None) or 'semantic_model'
        ax.scatter(row['avg_duration_per_question'], row['accuracy_percent'],
                   c=[colors[idx]], marker=sem_markers.get(sem, 'o'),
                   s=80, edgecolors='black', linewidth=1.5, zorder=3)
        ax.annotate(row['run_label'], (row['avg_duration_per_question'], row['accuracy_percent']),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)

    # Color bar to show chronological order
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Run order (older → newer)', fontsize=10)
    cbar.set_ticks([0, max(n - 1, 1)])
    cbar.set_ticklabels(['oldest', 'newest'])

    if len(unique_sem) > 1:
        import matplotlib.lines as mlines
        sem_handles = [mlines.Line2D([], [], color='grey', marker=sem_markers[s],
                                     linestyle='None', markersize=7, label=s) for s in unique_sem]
        ax.legend(handles=sem_handles, loc='upper right', fontsize=9, title='Semantic model')

    ax.set_xlabel('Avg Duration (seconds) - lower is better', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    llama_version = 'unknown'
    try:
        llama_version = 'b' + open(r'c:\llm\llama\version.txt').read().strip()
    except Exception:
        pass
    ax.set_title(f'{model_name} — History ({n} runs)\n(NVIDIA RTX 2000 Mobile, 4GB VRAM, llama.cpp {llama_version})',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 65, 5))
    ax.set_yticks(np.arange(0, 110, 10))
    ax.grid(True, alpha=0.3)
    ax.add_patch(plt.matplotlib.patches.Rectangle(
        (0, 90), 5, 10, transform=ax.transData,
        facecolor='#00cc44', alpha=0.12, zorder=0
    ))

    plt.tight_layout()
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    chart_path = os.path.join(output_dir, f'chart_history_{safe_name}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {chart_path}')
    plt.show()


def plot_tokens(model_name):
    """Bar chart of completion tokens per question for the latest run of a model."""
    import matplotlib.pyplot as plt
    import numpy as np
    import glob as _glob

    # Find latest log file for this model
    pattern = os.path.join(output_dir, 'log', f'*_{model_name}.json')
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f'No log files found for model: {model_name}')
        return
    log_path = files[-1]

    with open(log_path, 'r', encoding='utf-8') as fh:
        records = json.load(fh)

    records = sorted(records, key=lambda r: r.get('nbr', 0))
    labels = [f"Q{r['nbr']}" for r in records]
    values = [r.get('completion_tokens', 0) or 0 for r in records]
    colors = ['#1f77b4' for r in records]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.8, width=0.7)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Completion Tokens', fontsize=12, fontweight='bold')
    ts = os.path.basename(log_path)[:13]
    ax.set_title(f'Completion Tokens per Question — {model_name} ({ts})', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    tokens_chart_path = os.path.join(output_dir, 'chart_tokens.png')
    plt.savefig(tokens_chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {tokens_chart_path}')
    plt.show()


def plot_tokens_per_s(comparison_table):
    """Plot scatter chart of Tokens/s vs Accuracy. Only shows models with tokens_per_s data."""
    import matplotlib.pyplot as plt

    if 'avg_tokens_per_s' not in comparison_table.columns or not comparison_table['avg_tokens_per_s'].notna().any():
        print("No tokens/s data available yet.")
        return

    base_colors = ['#1f77b4', '#2e8b57', '#ff8c00', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    has_all_runs = 'timestamp' in comparison_table.columns

    df = comparison_table.dropna(subset=['avg_tokens_per_s'])

    plt.figure(figsize=(12, 7))

    if has_all_runs:
        unique_models = sorted(df['model'].unique())
        model_color = {m: base_colors[i % len(base_colors)] for i, m in enumerate(unique_models)}

        for model in unique_models:
            mdf = df[df['model'] == model].sort_values('timestamp')
            x = mdf['avg_tokens_per_s'].tolist()
            y = mdf['accuracy_percent'].tolist()
            plt.scatter(x, y, c=model_color[model], s=60, alpha=0.7, edgecolors='black', linewidth=1.5, label=model, zorder=3)

        plt.legend(loc='lower left', fontsize=10)
        n_label = f'{len(unique_models)} models, {len(df)} runs'
        all_tps = df['avg_tokens_per_s'].tolist()
    else:
        models = df['model'].tolist()
        accuracy = df['accuracy_percent'].tolist()
        tps = df['avg_tokens_per_s'].tolist()
        colors = (base_colors * ((len(models) // len(base_colors)) + 1))[:len(models)]
        plt.scatter(tps, accuracy, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=2)
        for i, model in enumerate(models):
            plt.annotate(model, (tps[i], accuracy[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
        n_label = f'{len(models)} models'
        all_tps = tps

    plt.xlabel('Avg Tokens/s - higher is better', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    llama_version = 'unknown'
    try:
        llama_version = 'b' + open(r'c:\llm\llama\version.txt').read().strip()
    except Exception:
        pass
    plt.title(f'Text to SQL: Tokens/s vs Accuracy ({n_label})\n(NVIDIA RTX 2000 Mobile, 4GB VRAM, 32GB RAM, llama.cpp {llama_version})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(all_tps) * 1.1 if all_tps and max(all_tps) > 0 else 10)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()


MALLOY_SYSTEM_PROMPT_PATH = r'c:\llm\system_prompts\malloy_prompts.txt'
MALLOY_MODEL_PATH         = r'c:\llm\semantic_models\semantic_model.malloy'
MALLOY_CLI                = r'C:\Users\mdjouallah\AppData\Roaming\npm\malloy-cli.cmd'

BORING_SYSTEM_PROMPT_PATH = r'c:\llm\system_prompts\boring_prompts.txt'
BORING_YAML_PATH          = r'c:\llm\semantic_models\semantic_model_boring.yaml'


def _malloy_system_prompt():
    with open(MALLOY_SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()


def _get_ai_malloy_query(user_message, model_name, endpoint=None):
    """Ask the LLM to generate a Malloy query."""
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': _malloy_system_prompt()},
                {'role': 'user',   'content': str(user_message)},
            ],
            'stream': False,
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        message = data.get('choices', [{}])[0].get('message', {})
        generated_text = message.get('content', '') or message.get('reasoning_content', '')

        timings = data.get('timings', {})
        usage   = data.get('usage', {})
        _token_accumulator["completion_tokens"] += usage.get('completion_tokens', 0) or timings.get('predicted_n', 0)
        _token_accumulator["generation_ms"]     += timings.get('predicted_ms', 0.0)

        if not generated_text:
            return f"Error: No content received. Response: {data}"
    except Exception as e:
        return f"Error: {e}"

    # Strip think tags and code fences
    text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    blocks = re.findall(r'```(?:malloy)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    if blocks:
        text = blocks[-1].strip()
    return text.strip()


def _execute_malloy_query(query_text):
    """Run a Malloy query via malloy-cli. Returns (df_or_error, query_text)."""
    import subprocess
    query_text = re.sub(r'^\s*run\s*:\s*', '', query_text).strip()
    query_text = ' '.join(query_text.split())
    try:
        result = subprocess.run(
            [MALLOY_CLI, 'run', MALLOY_MODEL_PATH, query_text],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return f"Malloy error: {result.stderr.strip()}", query_text
        clean = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        rows = json.loads(clean)
        return pd.DataFrame(rows), query_text
    except subprocess.TimeoutExpired:
        return "Malloy error: timeout", query_text
    except Exception as e:
        return f"Malloy error: {e}", query_text


def ask_question_malloy(questions_list, test_model):
    """Malloy variant: LLM generates Malloy queries executed via malloy-cli."""
    results_data = []
    for i, x in enumerate(questions_list):
        print(f"Question {i+1}: {x}")
        _token_accumulator["completion_tokens"] = 0
        _token_accumulator["generation_ms"] = 0.0
        start_time = time.time()
        llm_elapsed = 0.0
        exec_elapsed = 0.0

        t0 = time.time(); query_text = _get_ai_malloy_query(x, test_model); llm_elapsed += time.time() - t0
        error_details = None
        attempts_count = 1
        final_query = query_text
        query_result_data_json = []
        result_row_count = 0

        if query_text.startswith("Error"):
            error_details = f"AI Error: {query_text}"
        else:
            t0 = time.time(); result, final_query = _execute_malloy_query(query_text); exec_elapsed += time.time() - t0

            while isinstance(result, str) and attempts_count < max_attempts:
                attempts_count += 1
                retry_msg = (
                    f"The following Malloy query failed:\n{final_query}\n"
                    f"Error: {result}\n"
                    f"Fix the query and return ONLY the corrected Malloy query.\n"
                    f"Common mistakes to check:\n"
                    f"- order_by must use bare field names, NEVER dot notation: use 'i_product_name' not 'item.i_product_name'\n"
                    f"- group_by can use dot notation (item.i_product_name) but order_by cannot\n"
                    f"- query must be a single line"
                )
                t0 = time.time(); query_text = _get_ai_malloy_query(retry_msg, test_model); llm_elapsed += time.time() - t0
                t0 = time.time(); result, final_query = _execute_malloy_query(query_text); exec_elapsed += time.time() - t0

            if isinstance(result, pd.DataFrame):
                display(result)
                query_result_data_json = result.to_dict('records')
                result_row_count = len(result)
            else:
                print(f"Execution: FAILED — {result}")
                error_details = f"Execution Error: {result}"

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        completion_tokens = _token_accumulator["completion_tokens"]
        generation_s = round(_token_accumulator["generation_ms"] / 1000, 2)
        tokens_per_s = round(completion_tokens / generation_s, 1) if generation_s > 0 else None
        print(f" ### LLM: {llm_elapsed:.1f}s | Exec: {exec_elapsed:.1f}s | Total: {duration:.1f}s ###")
        results_data.append({
            "model":             test_model,
            "SF":                SF,
            "semantic_model":    "malloy",
            "timestamp":         timestamp,
            "nbr":               i + 1,
            "question":          x,
            "duration_s":        duration,
            "completion_tokens": completion_tokens,
            "generation_s":      generation_s,
            "tokens_per_s":      tokens_per_s,
            "sql_query":         final_query,
            "attempts":          attempts_count,
            "feedback_iterations": 0,
            "result":            query_result_data_json,
            "result_count":      result_row_count,
            "error_details":     error_details,
        })

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    sanitized_model = re.sub(r'[\\/*?:"<>|]', '_', test_model)
    output_filename = f"{timestamp}_{sanitized_model}_malloy.json"
    output_path = os.path.join(log_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    return f"Malloy run complete. Results saved to {output_path}"


# ---------------------------------------------------------------------------
# Boring Semantic Layer runtime
# ---------------------------------------------------------------------------

def _boring_system_prompt():
    with open(BORING_SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()


_boring_models_cache = None

def _load_boring_models():
    global _boring_models_cache
    if _boring_models_cache is not None:
        return _boring_models_cache
    import ibis
    from boring_semantic_layer import from_yaml
    ibis_con = ibis.duckdb.from_connection(con)
    tables = {name: ibis_con.table(name) for name in
              ("v_transactions", "date_dim", "store", "item", "customer")}
    _boring_models_cache = from_yaml(BORING_YAML_PATH, tables=tables)
    return _boring_models_cache


def _get_ai_boring_query(user_message, model_name, endpoint=None):
    """Ask the LLM to generate Boring Semantic Layer Python code."""
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': _boring_system_prompt()},
                {'role': 'user',   'content': str(user_message)},
            ],
            'stream': False,
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        message = data.get('choices', [{}])[0].get('message', {})
        generated_text = message.get('content', '') or message.get('reasoning_content', '')
        timings = data.get('timings', {})
        usage   = data.get('usage', {})
        _token_accumulator["completion_tokens"] += usage.get('completion_tokens', 0) or timings.get('predicted_n', 0)
        _token_accumulator["generation_ms"]     += timings.get('predicted_ms', 0.0)
        if not generated_text:
            return f"Error: No content received. Response: {data}"
    except Exception as e:
        return f"Error: {e}"
    text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    if blocks:
        text = blocks[-1].strip()
    return text.strip()


def _execute_boring_query(code_text, models):
    """Execute BSL Python code. Returns (df_or_error_str, code_text).

    LLM code must assign a BSL query (without .execute()) to `query`, plus any
    post-processing steps that operate on `result` after execution.
    We call .execute() ourselves so we can print the compiled SQL first.
    """
    import ibis
    ns = {
        "transactions": models["transactions"],
        "pd": pd,
    }
    try:
        exec(code_text, ns)  # noqa: S102
        query = ns.get("query")
        if query is None:
            return "Boring error: code did not assign to 'query'", code_text

        if isinstance(query, pd.DataFrame):
            return "Boring error: 'query' is already a DataFrame — assign the BSL expression before calling .execute()", code_text

        # Print compiled SQL before executing
        print(f"[SQL]\n{query.compile()}\n")

        result = query.execute()

        # Run any post-processing the LLM added (expects `result` in scope)
        post = ns.get("_post")
        if callable(post):
            result = post(result)

        if not isinstance(result, pd.DataFrame):
            return f"Boring error: 'result' is not a DataFrame after post-processing", code_text
        return result, code_text
    except Exception as e:
        return f"Boring error: {e}", code_text


def ask_question_boring(questions_list, test_model):
    """Boring Semantic Layer variant: LLM generates BSL Python code executed locally."""
    models = _load_boring_models()
    results_data = []
    for i, x in enumerate(questions_list):
        print(f"Question {i+1}: {x}")
        _token_accumulator["completion_tokens"] = 0
        _token_accumulator["generation_ms"] = 0.0
        start_time = time.time()
        llm_elapsed = 0.0
        exec_elapsed = 0.0

        t0 = time.time(); code_text = _get_ai_boring_query(x, test_model); llm_elapsed += time.time() - t0
        error_details = None
        attempts_count = 1
        final_code = code_text
        query_result_data_json = []
        result_row_count = 0

        if code_text.startswith("Error"):
            error_details = f"AI Error: {code_text}"
        else:
            t0 = time.time(); result, final_code = _execute_boring_query(code_text, models); exec_elapsed += time.time() - t0

            while isinstance(result, str) and attempts_count < max_attempts:
                attempts_count += 1
                retry_msg = (
                    f"The following Boring Semantic Layer Python code failed:\n{final_code}\n"
                    f"Error: {result}\n"
                    f"Fix the code and return ONLY the corrected Python code block.\n"
                    f"Remember: assign the final Ibis table expression to a variable named 'query'."
                )
                t0 = time.time(); code_text = _get_ai_boring_query(retry_msg, test_model); llm_elapsed += time.time() - t0
                t0 = time.time(); result, final_code = _execute_boring_query(code_text, models); exec_elapsed += time.time() - t0

            if isinstance(result, pd.DataFrame):
                display(result)
                query_result_data_json = result.to_dict('records')
                result_row_count = len(result)
            else:
                print(f"Execution: FAILED — {result}")
                error_details = f"Execution Error: {result}"

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        completion_tokens = _token_accumulator["completion_tokens"]
        generation_s = round(_token_accumulator["generation_ms"] / 1000, 2)
        tokens_per_s = round(completion_tokens / generation_s, 1) if generation_s > 0 else None
        print(f" ### LLM: {llm_elapsed:.1f}s | Exec: {exec_elapsed:.1f}s | Total: {duration:.1f}s ###")
        results_data.append({
            "model":             test_model,
            "SF":                SF,
            "semantic_model":    "boring",
            "timestamp":         timestamp,
            "nbr":               i + 1,
            "question":          x,
            "duration_s":        duration,
            "completion_tokens": completion_tokens,
            "generation_s":      generation_s,
            "tokens_per_s":      tokens_per_s,
            "sql_query":         final_code,
            "attempts":          attempts_count,
            "feedback_iterations": 0,
            "result":            query_result_data_json,
            "result_count":      result_row_count,
            "error_details":     error_details,
        })

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    sanitized_model = re.sub(r'[\\/*?:"<>|]', '_', test_model)
    output_filename = f"{timestamp}_{sanitized_model}_boring.json"
    output_path = os.path.join(log_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    return f"Boring run complete. Results saved to {output_path}"


def run_test(model_name, semantic_model=None):
    """Complete test workflow: start server, run tests, analyze, plot, stop server.

    semantic_model: 'malloy' for Malloy mode, 'boring' for Boring Semantic Layer mode, or path to a .txt system prompt for SQL mode.
    """
    global timestamp, last_tested_model, test_semantic_model_url
    last_tested_model = model_name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Malloy mode
    if semantic_model == 'malloy':
        try:
            start_server(model_name)
            ask_question_malloy(questions, model_name)
        finally:
            stop_server()
        return

    # Boring Semantic Layer mode
    if semantic_model == 'boring':
        try:
            start_server(model_name)
            ask_question_boring(questions, model_name)
        finally:
            stop_server()
        return

    if semantic_model is not None:
        test_semantic_model_url = semantic_model
    try:
        start_server(model_name)
        ask_question(questions, model_name)
    finally:
        stop_server()


def show_wrong_answers(model_name=None):
    """Display wrong answers for the latest run of a model, compared to the baseline."""
    import glob as _glob
    from IPython.display import display as _display

    if model_name is None:
        if last_tested_model is None:
            print("No model tested yet. Pass a model_name.")
            return
        model_name = last_tested_model

    baseline_files = sorted(_glob.glob(os.path.join(output_dir, 'log', f'*{model0}*.json')))
    baseline_records = json.load(open(baseline_files[-1], encoding='utf-8'))
    ref_map = {r['nbr']: r['result'] for r in baseline_records}

    model_files = sorted(_glob.glob(os.path.join(output_dir, 'log', f'*{model_name}*.json')))
    if not model_files:
        print(f"No log files found for model: {model_name}")
        return
    model_records = json.load(open(model_files[-1], encoding='utf-8'))
    run_ts = model_records[0]['timestamp']
    print(f"Model: {model_name}  |  Run: {run_ts}")
    print()

    wrong_count = 0
    for r in model_records:
        ref = ref_map.get(r['nbr'], [])
        got = r.get('result', [])
        status = _compare_stored_results(ref, got)
        if status == 'exact':
            continue

        wrong_count += 1
        ref_df = pd.DataFrame(ref) if ref else pd.DataFrame()
        got_df = pd.DataFrame(got) if got else pd.DataFrame()
        print('=' * 70)
        print(f"Q{r['nbr']}: {r['question']}")
        print(f"Attempts: {r['attempts']}  |  Status: {status}")
        print()
        print('Generated SQL:')
        print(r['sql_query'])
        print()
        from IPython.display import HTML as _HTML
        side_by_side = (
            f'<div style="display:flex;gap:2em">'
            f'<div><b>Expected ({len(ref_df)} rows)</b>{ref_df.head(10).to_html(index=False)}</div>'
            f'<div><b>Got ({len(got_df)} rows)</b>{got_df.head(10).to_html(index=False)}</div>'
            f'</div>'
        )
        _display(_HTML(side_by_side))
        print()

    if wrong_count == 0:
        print('All answers correct.')
