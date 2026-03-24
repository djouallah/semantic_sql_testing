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
from IPython.display import display

# ============================================
# Module-level state (set by init())
# ============================================
con = None
timestamp = None
questions = []
llama_server_process = None

# ============================================
# Config defaults (overridden from notebook)
# ============================================
LLAMA_CPP_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
TIMEOUT_SECONDS = 300
max_attempts = 3
SF = 0.1
output_dir = r"c:\llm"
test_semantic_model_url = r'c:\llm\semantic_model.txt'
questions_url = r'c:\llm\questions.json'
model0 = "claude-opus-4-6"
enable_feedback_loop = True
MODEL_CONFIGS = {}


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


def init():
    """Initialize database connection, timestamp, and load questions."""
    global con, timestamp, questions

    if SF < 1:
        schema = f"DS{str(SF).replace('.', '_')}"
    else:
        schema = f'DS{SF:02d}'

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, f"{schema}.duckdb")

    if not pathlib.Path(db_path).exists():
        c = duckdb.connect(db_path)
        c.sql("SET memory_limit = '14GB' ")
        c.sql(f"CALL dsdgen(sf={SF})")
        c.close()

    con = duckdb.connect()
    con.sql(f"ATTACH '{db_path}' AS ds (READ_ONLY); USE ds;")
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


def start_server(model_name):
    """Start llama-server with the specified model configuration."""
    global llama_server_process

    if model_name not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

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
# LLM interaction
# ============================================
def get_ai_response(user_message, model_name, endpoint=None):
    """Generate SQL query using llama.cpp server."""
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    user_message = str(user_message)

    system_prompt = ""
    try:
        github_response = requests.get(test_semantic_model_url)
        github_response.raise_for_status()
        system_prompt = github_response.text.strip()
    except requests.RequestException:
        try:
            with open(test_semantic_model_url, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            return f"Error loading system prompt: {e}"

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


def get_feedback_response(original_question, sql_query, query_results, model_name, endpoint=None):
    """Get LLM feedback on query results for improvement."""
    if endpoint is None:
        endpoint = LLAMA_CPP_ENDPOINT
    original_question = str(original_question)

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

    feedback_prompt = f"""
You previously generated this SQL query for the following question:

ORIGINAL QUESTION: {original_question}

YOUR SQL QUERY:
{sql_query}

QUERY RESULTS:
{results_preview}{quality_section}

Please review your SQL query and its results carefully. Does the output correctly answer the original question?

Pay special attention to:
- Are there duplicate rows when the question asks for distinct items/categories?
- Is the data properly ordered as requested (alphabetically, numerically, etc.)?
- Does the result make logical sense for the question asked?
- Are you selecting the right columns and using appropriate filtering?

CRITICAL: If you see duplicate values in a single column when the question asks for "different" or "various" items, this is almost always WRONG. Use DISTINCT to get unique values only.

If the results look correct and fully answer the question, respond with: "RESULTS_CORRECT"

If the results need improvement (wrong data, missing information, incorrect calculations, duplicates, wrong ordering, etc.), provide a revised SQL query that better answers the question.

IMPORTANT: Return ONLY the SQL query without any explanation, markdown formatting, or additional text. Just the pure SQL statement.

Your response:"""

    try:
        github_response = requests.get(test_semantic_model_url)
        github_response.raise_for_status()
        system_prompt = github_response.text.strip()
    except requests.RequestException:
        try:
            with open(test_semantic_model_url, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            return f"Error loading system prompt: {e}"

    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': feedback_prompt}
            ],
            'stream': False
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        message = data.get('choices', [{}])[0].get('message', {})
        generated_text = message.get('content', '') or message.get('reasoning_content', '')

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
def execute_sql_with_retry(query, test_model):
    """Execute SQL with retry on syntax errors."""
    attempt = 1
    current_query = query.strip()

    while attempt <= max_attempts:
        con.sql("SET enable_progress_bar_print = false")
        con.sql("SET progress_bar_time = 0")

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
                return f"Query execution timed out after {TIMEOUT_SECONDS} seconds.", attempt, "query runs forever"
            time.sleep(0.1)

        if result_container["error"]:
            error_message = result_container["error"].lower()
            if "syntax" not in error_message and "parser" not in error_message and "binder" not in error_message:
                return f"Non-syntax error: {result_container['error']}", attempt, current_query

            print(current_query)
            print(f"Attempt {attempt}/{max_attempts} failed with syntax error: {result_container['error']}")

            if attempt == max_attempts:
                return f"Max attempts reached. Last error: {result_container['error']}", attempt, current_query

            message = (
                f"The following SQL query has a syntax error: '{current_query}'.\n"
                f"Error message: {result_container['error']}\n"
                f"Please provide the corrected SQL query. Return only the corrected query without explanation."
            )

            corrected_query = get_ai_response(message, test_model)

            if corrected_query.startswith("Error"):
                return f"Failed to get corrected query: {corrected_query}", attempt, current_query

            current_query = corrected_query.strip()
            attempt += 1
        else:
            return result_container["result"], attempt, current_query

    return "Unexpected error or loop termination", attempt, current_query


# ============================================
# Test runner
# ============================================
def ask_question(questions_list, test_model):
    """Process questions and generate SQL queries using llama.cpp server."""
    results_data = []
    for i, x in enumerate(questions_list):
        print(f"Question {i+1}: {x}")
        start_time = time.time()
        sql_query_or_error = get_ai_response(x, test_model)
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
            result_from_execution, attempts_count, query_returned = execute_sql_with_retry(sql_query_or_error, test_model)
            final_sql_query = query_returned

            is_successful = isinstance(result_from_execution, pd.DataFrame)

            if is_successful and enable_feedback_loop and feedback_iterations < max_attempts:
                current_query = query_returned
                current_result = result_from_execution

                while feedback_iterations < max_attempts:
                    feedback_response = get_feedback_response(x, current_query, current_result, test_model)

                    if "RESULTS_CORRECT" in feedback_response.upper():
                        break
                    elif feedback_response.startswith("Error"):
                        break
                    else:
                        revised_result, revised_attempts, revised_query = execute_sql_with_retry(feedback_response, test_model)

                        if isinstance(revised_result, pd.DataFrame):
                            current_query = revised_query
                            current_result = revised_result
                            final_sql_query = revised_query
                            attempts_count = (attempts_count or 0) + (revised_attempts or 0)
                        else:
                            break

                    feedback_iterations += 1

                result_from_execution = current_result

            display(result_from_execution)
            is_successful = isinstance(result_from_execution, pd.DataFrame)

            if is_successful:
                query_result_data_json = result_from_execution.to_dict('records')
                error_details = None
                result_row_count = len(result_from_execution)
            else:
                print("Execution: FAILED")
                error_details = f"Execution Error: {result_from_execution}"
                result_row_count = 0

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f" ############################### ")
        results_data.append({
            "model": test_model,
            "SF": SF,
            "timestamp": timestamp,
            "nbr": i + 1,
            "question": x,
            "duration_s": duration,
            "sql_query": final_sql_query,
            "attempts": attempts_count,
            "feedback_iterations": feedback_iterations,
            "result": query_result_data_json,
            "result_count": result_row_count,
            "error_details": error_details
        })

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


def plot_results(comparison_table):
    """Plot scatter chart of Speed vs Accuracy and save to chart.png."""
    import matplotlib.pyplot as plt

    models = comparison_table['model'].tolist()
    accuracy = comparison_table['accuracy_percent'].tolist()
    avg_time = comparison_table['avg_duration_per_question'].tolist()

    plt.figure(figsize=(10, 6))

    base_colors = ['#1f77b4', '#2e8b57', '#ff8c00', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = (base_colors * ((len(models) // len(base_colors)) + 1))[:len(models)]

    plt.scatter(avg_time, accuracy, c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=2)

    for i, model in enumerate(models):
        plt.annotate(model, (avg_time[i], accuracy[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.xlabel('Avg Duration (seconds) - lower is better', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    llama_version = 'unknown'
    try:
        llama_version = 'b' + open(r'c:\llm\llama\version.txt').read().strip()
    except Exception:
        pass
    plt.title(f'Text to SQL: Duration vs Accuracy ({len(models)} models)\n(NVIDIA RTX 2000 Mobile, 4GB VRAM, 32GB RAM, llama.cpp {llama_version})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(avg_time) * 1.1 if avg_time and max(avg_time) > 0 else 10)
    plt.ylim(0, 105)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {chart_path}')
    plt.show()


def run_test(model_name):
    """Complete test workflow: start server, run tests, analyze, plot, stop server."""
    global timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        start_server(model_name)
        ask_question(questions, model_name)
    finally:
        stop_server()
