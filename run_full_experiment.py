#!/usr/bin/env python3
"""
Run full LLM experiment with robust retry, checkpointing, and detailed logging.

Features:
- pinned model version
- deterministic sampling (temperature=0, top_p=1.0)
- exponential backoff with full jitter and max wait
- JSONL API log with usage and response metadata
- CSV checkpoint every `checkpoint_interval` rows with response_id and status
- CLI flags: --simulate, --start-index, --batch-size, --dry-run

Usage:
  python run_full_experiment.py --dataset data/full_dataset.csv

Before running: set OPENAI_API_KEY in environment or .env file (do NOT commit .env).
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def setup_args():
    p = argparse.ArgumentParser(description="Run full LLM experiment with checkpointing and logging")
    p.add_argument("--dataset", required=True, help="Path to input CSV (must include 'id' and 'text' columns)")
    p.add_argument("--output-dir", default="results", help="Directory to write outputs and logs")
    p.add_argument("--artifact-prefix", default="full", help="Prefix for output artifacts (e.g. full or pilot)")
    p.add_argument("--model", default="gpt-4o-2024-11-20", help="Model id to call (pinned recommended)")
    p.add_argument("--start-index", type=int, default=0, help="Start processing at this row index")
    p.add_argument("--checkpoint-interval", type=int, default=50, help="Save checkpoint every N rows")
    p.add_argument("--max-retries", type=int, default=8, help="Max API retry attempts")
    p.add_argument("--timeout", type=int, default=60, help="Per-request timeout (seconds)")
    p.add_argument("--dry-run", action="store_true", help="Do not call API; simulate responses")
    p.add_argument("--simulate", action="store_true", help="Return canned responses for offline testing")
    p.add_argument("--batch-size", type=int, default=1, help="Process N items per loop (1 = sequential)")
    p.add_argument("--prompt-profile", default="full", choices=["full", "pilot"], help="Prompt template profile to use")
    return p.parse_args()


def init_paths(output_dir: str, artifact_prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{artifact_prefix}_llm_output_checkpoint.csv")
    final_path = os.path.join(output_dir, f"{artifact_prefix}_llm_output.csv")
    jsonl_log = os.path.join(output_dir, f"{artifact_prefix}_api_log.jsonl")
    txt_log = os.path.join(output_dir, f"{artifact_prefix}_api_log.txt")
    return checkpoint_path, final_path, jsonl_log, txt_log


def openai_client_from_env():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise ValueError("OPENAI_API_KEY not set or placeholder. Set a valid key in env or .env file.")
    if OpenAI is None:
        raise RuntimeError("openai SDK not available. Install with: pip install openai")
    return OpenAI(api_key=api_key)


def cost_from_usage(prompt_tokens: int, completion_tokens: int) -> float:
    # Pricing assumptions (adjust if you have an up-to-date price table)
    # GPT-4o example: $2.5 / 1M input tokens, $10 / 1M output tokens
    return (prompt_tokens * 2.5 / 1_000_000.0) + (completion_tokens * 10.0 / 1_000_000.0)


def make_messages(text: str, prompt_profile: str):
    if prompt_profile == "pilot":
        system = (
            "You are a QA expert. Generate a BDD scenario for this User Story.\n"
            "User Story: {user_story}\n\n"
            "Use the following format:\n"
            "Scenario: [Scenario name]\n"
            "Given [precondition]\n"
            "When [action]\n"
            "Then [expected result]\n\n"
            "Make sure to:\n"
            "1. Use clear and specific steps.\n"
            "2. Include all necessary preconditions.\n"
            "3. Describe the main action clearly.\n"
            "4. Specify concrete expected results.\n"
            "5. Use business language.\n"
            "6. Do not include any explanations or multiple scenarios.\n"
            "7. Focus on the most important happy path scenario.\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"User Story: {text}"},
        ]

    system = (
        "You are a QA expert. Generate a BDD scenario for this User Story.\n"
        "Use the following format:\nScenario: [Scenario name]\nGiven [precondition]\nWhen [action]\nThen [expected result]\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def call_openai_with_retry(
    client,
    model: str,
    messages: list,
    max_retries: int = 8,
    timeout: int = 60,
    base_backoff: float = 1.0,
    max_wait: float = 60.0,
) -> Tuple[str, dict]:
    """Call OpenAI chat completions with exponential backoff + full jitter.

    Returns (text, meta_dict)
    """
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        start = time.time()
        ts = datetime.utcnow().isoformat() + "Z"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=timeout,
            )

            usage = getattr(resp, "usage", None) or {}
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

            cost = cost_from_usage(prompt_tokens, completion_tokens)

            # response text
            text = ""
            try:
                text = resp.choices[0].message.content.strip()
            except Exception:
                text = resp.choices[0].message.get("content", "").strip() if resp.choices else ""

            meta = {
                "timestamp": ts,
                "model": model,
                "response_model": getattr(resp, "model", model),
                "latency_s": time.time() - start,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "response_id": getattr(resp, "id", None),
                "status": "success",
            }
            return text, meta

        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            # classify rate limit-ish errors
            is_rate = any(x in err_text for x in ("rate_limit", "429", "too many requests", "quota"))
            if not is_rate:
                # non-retryable -> return INVALID with meta
                meta = {
                    "timestamp": ts,
                    "model": model,
                    "response_model": model,
                    "latency_s": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "response_id": None,
                    "status": "error",
                    "error": str(e),
                }
                return "INVALID", meta

            # retryable: exponential backoff with full jitter
            wait = min(max_wait, base_backoff * (2 ** attempt))
            jitter = random.random() * wait
            sleep_for = jitter
            attempt += 1
            if attempt > max_retries:
                break
            print(f"⚠️ Rate limit or transient error (attempt {attempt}/{max_retries}). Sleeping {sleep_for:.1f}s and retrying...")
            time.sleep(sleep_for)

    # exhausted
    ts = datetime.utcnow().isoformat() + "Z"
    meta = {
        "timestamp": ts,
        "model": model,
        "response_model": model,
        "latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "response_id": None,
        "status": "invalid_max_retries",
        "error": str(last_err),
    }
    return "INVALID", meta


def save_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return pd.read_csv(checkpoint_path)
    except Exception:
        return None


def main():
    args = setup_args()
    checkpoint_path, final_path, jsonl_log, txt_log = init_paths(args.output_dir, args.artifact_prefix)

    # Read dataset
    df = pd.read_csv(args.dataset, dtype=str)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Input CSV must contain 'id' and 'text' columns")

    # load checkpoint if present so resume can continue from already saved rows
    checkpoint_df = load_checkpoint(checkpoint_path)
    if checkpoint_df is not None:
        processed = len(checkpoint_df)
        start_idx = max(args.start_index, processed)
        results = checkpoint_df['llm_output'].astype(str).tolist()
        resp_ids = checkpoint_df.get('response_id', [None] * processed)
        statuses = checkpoint_df.get('status', [None] * processed)
        if hasattr(resp_ids, "tolist"):
            resp_ids = resp_ids.tolist()
        if hasattr(statuses, "tolist"):
            statuses = statuses.tolist()
        cumulative_cost = float(checkpoint_df.get('cumulative_cost_usd', 0.0).iloc[-1]) if 'cumulative_cost_usd' in checkpoint_df.columns else 0.0
        print(f"🔄 Found checkpoint with {processed} rows. Resuming at index {start_idx}.")
    else:
        start_idx = args.start_index
        results = []
        resp_ids = []
        statuses = []
        cumulative_cost = 0.0

    # prepare client
    client = None
    if not args.dry_run and not args.simulate:
        client = openai_client_from_env()

    total = len(df)
    print(f"🚀 Starting experiment: processing {total} rows (starting at {start_idx})")

    # open text log header
    if not os.path.exists(txt_log):
        with open(txt_log, "w", encoding="utf-8") as fh:
            fh.write("timestamp | model | latency | cost_usd | status | id\n")

    for i in range(start_idx, total, args.batch_size):
        row = df.iloc[i]
        uid = row['id']
        text = row['text']

        prompt_hash = md5(text)

        if args.simulate:
            out_text = f"SIMULATED_RESPONSE for id={uid}"
            meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "response_id": None,
                "status": "simulated",
            }
        elif args.dry_run:
            out_text = f"DRY_RUN for id={uid}"
            meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "response_id": None,
                "status": "dry_run",
            }
        else:
            messages = make_messages(text, args.prompt_profile)
            out_text, meta = call_openai_with_retry(
                client,
                model=args.model,
                messages=messages,
                max_retries=args.max_retries,
                timeout=args.timeout,
            )

        # accumulate
        results.append(out_text)
        resp_ids.append(meta.get('response_id'))
        statuses.append(meta.get('status'))
        cumulative_cost += float(meta.get('cost_usd', 0.0) or 0.0)

        # enriched meta to save
        meta_record = {
            "id": uid,
            "prompt_hash": prompt_hash,
            **meta,
            "input_id": uid,
            "cumulative_cost_usd": cumulative_cost,
            "prompt_preview": text[:200],
        }

        save_jsonl(jsonl_log, meta_record)

        # append a short line to human-readable log
        with open(txt_log, "a", encoding="utf-8") as fh:
            fh.write(f"{meta_record['timestamp']} | {meta_record.get('response_model', meta_record['model'])} | ${meta_record['cost_usd']:.6f} | {uid}\n")

        # checkpoint every N rows or at end
        if ((i + 1) % args.checkpoint_interval == 0) or (i + 1 == total):
            slice_df = df.iloc[:i+1].copy()
            slice_df['llm_output'] = results[:i+1]
            slice_df['response_id'] = resp_ids[:i+1]
            slice_df['status'] = statuses[:i+1]
            slice_df['cumulative_cost_usd'] = cumulative_cost
            slice_df.to_csv(checkpoint_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
            print(f"💾 Checkpoint saved: {i+1}/{total} rows. cumulative_cost=${cumulative_cost:.6f}")

        # short sleep to avoid aggressive throttling
        time.sleep(0.2)

    # write final output
    df_out = df.copy()
    df_out['llm_output'] = results
    df_out['response_id'] = resp_ids
    df_out['status'] = statuses
    df_out['cumulative_cost_usd'] = cumulative_cost
    df_out.to_csv(final_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    print(f"\n🎉 Experiment complete. Results: {final_path}")
    print(f"API JSONL log: {jsonl_log}")
    print(f"Human log: {txt_log}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Run full LLM experiment with robust retry, checkpointing, and detailed logging.

Features:
- pinned model version
- deterministic sampling (temperature=0, top_p=1.0)
- exponential backoff with full jitter and max wait
- JSONL API log with usage and response metadata
- CSV checkpoint every `checkpoint_interval` rows with response_id and status
- CLI flags: --simulate, --start-index, --batch-size, --dry-run

Usage:
  python run_full_experiment.py --dataset data/full_dataset.csv

Before running: set OPENAI_API_KEY in environment or .env file (do NOT commit .env).
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def setup_args():
    p = argparse.ArgumentParser(description="Run full LLM experiment with checkpointing and logging")
    p.add_argument("--dataset", required=True, help="Path to input CSV (must include 'id' and 'text' columns)")
    p.add_argument("--output-dir", default="results", help="Directory to write outputs and logs")
    p.add_argument("--model", default="gpt-4o-2024-11-20", help="Model id to call (pinned recommended)")
    p.add_argument("--start-index", type=int, default=0, help="Start processing at this row index")
    p.add_argument("--checkpoint-interval", type=int, default=50, help="Save checkpoint every N rows")
    p.add_argument("--max-retries", type=int, default=8, help="Max API retry attempts")
    p.add_argument("--timeout", type=int, default=60, help="Per-request timeout (seconds)")
    p.add_argument("--dry-run", action="store_true", help="Do not call API; simulate responses")
    p.add_argument("--simulate", action="store_true", help="Return canned responses for offline testing")
    p.add_argument("--batch-size", type=int, default=1, help="Process N items per loop (1 = sequential)")
    return p.parse_args()


def init_paths(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "full_llm_output_checkpoint.csv")
    final_path = os.path.join(output_dir, "full_llm_output.csv")
    jsonl_log = os.path.join(output_dir, "full_api_log.jsonl")
    txt_log = os.path.join(output_dir, "full_api_log.txt")
    return checkpoint_path, final_path, jsonl_log, txt_log


def openai_client_from_env():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise ValueError("OPENAI_API_KEY not set or placeholder. Set a valid key in env or .env file.")
    if OpenAI is None:
        raise RuntimeError("openai SDK not available. Install with: pip install openai")
    return OpenAI(api_key=api_key)


def cost_from_usage(prompt_tokens: int, completion_tokens: int) -> float:
    # Pricing assumptions (adjust if you have an up-to-date price table)
    # GPT-4o example: $2.5 / 1M input tokens, $10 / 1M output tokens
    return (prompt_tokens * 2.5 / 1_000_000.0) + (completion_tokens * 10.0 / 1_000_000.0)


def call_openai_with_retry(
    client,
    model: str,
    messages: list,
    max_retries: int = 8,
    timeout: int = 60,
    base_backoff: float = 1.0,
    max_wait: float = 60.0,
) -> Tuple[str, dict]:
    """Call OpenAI chat completions with exponential backoff + full jitter.

    Returns (text, meta_dict)
    """
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        start = time.time()
        ts = datetime.utcnow().isoformat() + "Z"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                timeout=timeout,
            )

            usage = getattr(resp, "usage", None) or {}
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

            cost = cost_from_usage(prompt_tokens, completion_tokens)

            # response text
            text = ""
            try:
                text = resp.choices[0].message.content.strip()
            except Exception:
                text = resp.choices[0].message.get("content", "").strip() if resp.choices else ""

            meta = {
                "timestamp": ts,
                "model": model,
                "latency_s": time.time() - start,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "response_id": getattr(resp, "id", None),
                "status": "success",
            }
            return text, meta

        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            # classify rate limit-ish errors
            is_rate = any(x in err_text for x in ("rate_limit", "429", "too many requests", "quota"))
            if not is_rate:
                # non-retryable -> return INVALID with meta
                meta = {
                    "timestamp": ts,
                    "model": model,
                    "latency_s": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "response_id": None,
                    "status": "error",
                    "error": str(e),
                }
                return "INVALID", meta

            # retryable: exponential backoff with full jitter
            wait = min(max_wait, base_backoff * (2 ** attempt))
            jitter = random.random() * wait
            sleep_for = jitter
            attempt += 1
            if attempt > max_retries:
                break
            print(f"⚠️ Rate limit or transient error (attempt {attempt}/{max_retries}). Sleeping {sleep_for:.1f}s and retrying...")
            time.sleep(sleep_for)

    # exhausted
    ts = datetime.utcnow().isoformat() + "Z"
    meta = {
        "timestamp": ts,
        "model": model,
        "latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "response_id": None,
        "status": "invalid_max_retries",
        "error": str(last_err),
    }
    return "INVALID", meta


def save_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return pd.read_csv(checkpoint_path)
    except Exception:
        return None


def main():
    args = setup_args()
    checkpoint_path, final_path, jsonl_log, txt_log = init_paths(args.output_dir)

    # Read dataset
    df = pd.read_csv(args.dataset, dtype=str)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Input CSV must contain 'id' and 'text' columns")

    # load checkpoint if present and start-index not explicitly set
    checkpoint_df = load_checkpoint(checkpoint_path)
    if checkpoint_df is not None and args.start_index == 0:
        processed = len(checkpoint_df)
        start_idx = processed
        results = checkpoint_df['llm_output'].astype(str).tolist()
        resp_ids = checkpoint_df.get('response_id', [None] * processed)
        statuses = checkpoint_df.get('status', [None] * processed)
        cumulative_cost = float(checkpoint_df.get('cumulative_cost_usd', 0.0).iloc[-1]) if 'cumulative_cost_usd' in checkpoint_df.columns else 0.0
        print(f"🔄 Found checkpoint with {processed} rows. Resuming at index {start_idx}.")
    else:
        start_idx = args.start_index
        results = []
        resp_ids = []
        statuses = []
        cumulative_cost = 0.0

    # prepare client
    client = None
    if not args.dry_run and not args.simulate:
        client = openai_client_from_env()

    # text -> prompt template
    def make_messages(text: str):
        system = (
            "You are a QA expert. Generate a BDD scenario for this User Story.\n"
            "Use the following format:\nScenario: [Scenario name]\nGiven [precondition]\nWhen [action]\nThen [expected result]\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

    total = len(df)
    print(f"🚀 Starting experiment: processing {total} rows (starting at {start_idx})")

    # open text log header
    if not os.path.exists(txt_log):
        with open(txt_log, "w", encoding="utf-8") as fh:
            fh.write("timestamp | model | latency | cost_usd | status | id\n")

    for i in range(start_idx, total, args.batch_size):
        row = df.iloc[i]
        uid = row['id']
        text = row['text']

        prompt_hash = md5(text)

        if args.simulate:
            out_text = f"SIMULATED_RESPONSE for id={uid}"
            meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "response_id": None,
                "status": "simulated",
            }
        elif args.dry_run:
            out_text = f"DRY_RUN for id={uid}"
            meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": args.model,
                "latency_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "response_id": None,
                "status": "dry_run",
            }
        else:
            messages = make_messages(text)
            out_text, meta = call_openai_with_retry(
                client,
                model=args.model,
                messages=messages,
                max_retries=args.max_retries,
                timeout=args.timeout,
            )

        # accumulate
        results.append(out_text)
        resp_ids.append(meta.get('response_id'))
        statuses.append(meta.get('status'))
        cumulative_cost += float(meta.get('cost_usd', 0.0) or 0.0)

        # enriched meta to save
        meta_record = {
            "id": uid,
            "prompt_hash": prompt_hash,
            **meta,
            "input_id": uid,
            "cumulative_cost_usd": cumulative_cost,
            "prompt_preview": text[:200],
        }

        save_jsonl(jsonl_log, meta_record)

        # append a short line to human-readable log
        with open(txt_log, "a", encoding="utf-8") as fh:
            fh.write(f"{meta_record['timestamp']} | {meta_record['model']} | {meta_record['latency_s']:.2f}s | ${meta_record['cost_usd']:.6f} | {meta_record['status']} | {uid}\n")

        # checkpoint every N rows or at end
        if ((i + 1) % args.checkpoint_interval == 0) or (i + 1 == total):
            slice_df = df.iloc[:i+1].copy()
            slice_df['llm_output'] = results
            slice_df['response_id'] = resp_ids
            slice_df['status'] = statuses
            slice_df['cumulative_cost_usd'] = cumulative_cost
            slice_df.to_csv(checkpoint_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
            print(f"💾 Checkpoint saved: {i+1}/{total} rows. cumulative_cost=${cumulative_cost:.6f}")

        # short sleep to avoid aggressive throttling
        time.sleep(0.2)

    # write final output
    df_out = df.copy()
    df_out['llm_output'] = results
    df_out['response_id'] = resp_ids
    df_out['status'] = statuses
    df_out['cumulative_cost_usd'] = cumulative_cost
    df_out.to_csv(final_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    print(f"\n🎉 Experiment complete. Results: {final_path}")
    print(f"API JSONL log: {jsonl_log}")
    print(f"Human log: {txt_log}")


if __name__ == '__main__':
    main()
