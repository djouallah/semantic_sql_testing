# SLM SQL Benchmark — Project Guide

## What This Is
A benchmarking framework that evaluates Small Language Models (SLMs) on natural language to SQL generation. Models run locally via llama.cpp against a TPC-DS star schema in DuckDB. 20 curated questions ranging from simple aggregations to complex cross-fact queries.

## Hardware
- GPU: NVIDIA RTX A2000 Laptop — **4GB VRAM only**
- RAM: 32GB system RAM
- Models above ~4GB Q4_K_M will overflow VRAM and run painfully slow via Windows VMM
- Practical limit: 4B parameter models at Q4_K_M

## Key Files
- `slm_benchmark.py` — core module (~1050 lines), all logic lives here
- `SLM_SQL_test.ipynb` — main notebook, run tests from here
- `models_config.json` — model registry (path, hf_repo, sampling params)
- `semantic_model.txt` — system prompt sent to LLMs (schema + business rules)
- `questions.json` — 20 test questions
- `DS01.duckdb` — TPC-DS SF=1 database (ignored by git)
- `log/` — JSON result files, one per model run
- `build_benchmark.py` — internal tool for schema introspection and baseline generation

## Database Schema
Star schema with 2 facts and 4 dimensions:

**Facts:** `store_sales` (2.88M rows), `store_returns` (288K rows)
**Dims:** `date_dim`, `store`, `customer`, `item`

### Critical: No Fact-to-Fact Joins
Never join `store_sales` and `store_returns` directly. Always aggregate each fact separately in CTEs, then FULL OUTER JOIN the results. Violating this causes row multiplication.

### Key Measures
- `total_sales = SUM(ss_sales_price * ss_quantity)` — from store_sales
- `total_returns = SUM(sr_return_amt)` — from store_returns
- `net_sales` and `return_rate` — require the CTE + FULL OUTER JOIN pattern

## How a Benchmark Run Works
1. `bench.load_config('./models_config.json')` — loads model configs
2. `bench.update_llama_cpp()` — downloads latest llama.cpp binary
3. `bench.run_test('model-name')` — starts llama-server, runs 20 questions, stops server
   - LLM gets `semantic_model.txt` as system prompt
   - On SQL error: retries up to 3 times with error feedback
   - If `enable_feedback_loop: true`: LLM sees result preview and can self-correct
   - Timeout: 60 seconds per question
4. `bench.analyze_all_runs()` — loads all logs, compares against Claude Opus baseline
5. `bench.plot_results(all_runs)` — generates chart (duration vs accuracy, x-axis capped at 60s)

## Model Config Fields
```json
{
  "model_path": "C:/llm/models/Model-Q4_K_M.gguf",
  "hf_repo": "unsloth/Model-GGUF",       // auto-downloaded if missing
  "context_size": 8192,
  "gpu_layers": 99,                        // 99 = full GPU offload
  "reasoning_budget": 0,                   // 0 = disable thinking (Qwen3 style)
  "flash_attn": "on",
  "fit": true,                             // optional: auto-fit context to VRAM
  "batch_size": 4096,
  "temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.05
}
```

## Evaluation
- Baseline: Claude Opus 4.6 (runs via API, results stored in log/)
- Accuracy: exact result match (row count + values, 0.5% numeric tolerance, column order ignored)
- `accuracy_percent` — after feedback loop
- `raw_accuracy_percent` — first attempt only
- Chart: X = avg duration per question, Y = accuracy %, both axes capped at hardware limits

## Adding a New Model
Add an entry to `models_config.json`. The script downloads the model automatically via `hf_repo`. Use Q4_K_M quant — anything larger than ~3.5GB will exceed the 4GB VRAM budget.

## Llama.cpp Server
Runs on `http://127.0.0.1:8080`. Log goes to `llama_server.log` (git-ignored). If the port is stuck, `bench.kill_process_on_port(8080)` clears it.
