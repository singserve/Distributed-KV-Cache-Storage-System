# language: python
import random
from pathlib import Path
import sys
import json
import time
import argparse
import requests



prefill_endpoints = ["10.0.13.1:8101", "10.0.13.1:8100","10.0.13.1:8102","10.0.13.1:8103"]   # GPU endpoints for prefill
decode_endpoints = ["10.0.14.1:8201", "10.0.14.1:8200","10.0.14.1:8202","10.0.14.1:8203"]    # GPU endpoints for decode

def make_prefix(index):
    # make long prefix for prefix cache
    return f"Shared context #{index}: " + "This is long context. " * 20

def make_suffix(index):
    return f"Question {index}: Summarize the context."

def pick_endpoints(index, strategy, pre, dec):
    if strategy == "Designate":
        prefill = pre
        decode = dec
    elif strategy == "RR":
        prefill = prefill_endpoints[index%len(prefill_endpoints)]
        decode = random.choice(decode_endpoints)
    elif strategy == "Random":
        prefill = random.choice(prefill_endpoints)
        decode = random.choice(decode_endpoints)
    else:
        prefill = random.choice(prefill_endpoints)
        decode = random.choice(decode_endpoints)
    return prefill, decode

def build_record(prompt, prefill, decode, model, max_tokens):
    user_field = f"prefill={prefill};decode={decode}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "user": user_field,
    }


def generate_input_file(args):

    out = Path(args.datapath)
    out.unlink(missing_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for index in range(args.request_num):
            prefix = make_prefix(index)
            suffix = make_suffix(index)
            prompt = prefix + suffix
            prefill, decode = pick_endpoints(index, args.strategy, args.prefill, args.decode)
            rec = build_record(prompt, prefill, decode, args.model, args.max_tokens)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {args.datapath} ({args.request_num} records) using strategy={args.strategy}")


def send_with_openai(record, api_base,client):
    if client is None:
        raise RuntimeError("openai package not installed")
    client.api_base = api_base.rstrip("/") + "/v1"
    # Completion API  can be replaced by ChatCompletion.create
    return client.Completion.create(
        model=record.get("model"),
        prompt=record.get("prompt"),
        max_tokens=record.get("max_tokens", 16),
        user=record.get("user")
    )

def send_with_requests(record, api_base):
    url = api_base.rstrip("/") + "/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": record.get("model"),
        "prompt": record.get("prompt"),
        "max_tokens": record.get("max_tokens", 16),
        "user": record.get("user"),
    }
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

def parse_metrics_text(metrics_text):
    hits = 0
    queries = 0
    lines = []
    for line in metrics_text.splitlines():
        lines.append(line)
        if line.startswith("vllm:prefix_cache_hits_total"):
            try:
                hits += int(float(line.split()[-1]))
            except Exception:
                pass
        elif line.startswith("vllm:prefix_cache_queries_total"):
            try:
                queries += int(float(line.split()[-1]))
            except Exception:
                pass
    return hits, queries, lines

def show_metrics(args):
    metrics_url = args.metric_url.rstrip("/") + "/metrics"
    try:
        r = requests.get(metrics_url)
        r.raise_for_status()
        hits, queries, raw_lines = parse_metrics_text(r.text)
        print("\n--- metrics (filtered lines) ---")
        for l in raw_lines:
            if l.startswith("vllm:prefix_cache"):
                print(l)
        print("\nsummary:")
        print("prefix_cache_hits_total =", hits)
        print("prefix_cache_queries_total =", queries)
        if queries:
            print("hit_rate =", round(hits / queries, 6))
        else:
            print("hit_rate = N/A (no queries)")
    except Exception as e:
        print("failed to fetch metrics from", metrics_url, ":", e, file=sys.stderr)
        sys.exit(1)

def send_to_proxy_server(args):

    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        client = None

    with open(args.datapath, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    successes = 0
    failures = 0
    for rec in records:
        try:
            if args.use_requests or client is None:
                resp = send_with_requests(rec, args.proxy)
            else:
                resp = send_with_openai(rec, args.proxy,client)
            successes += 1
        except Exception as e:
            failures += 1
            print("request failed:", rec.get("id", "<no-id>"), e)
        time.sleep(args.delay)

    print(f"sent {len(records)} requests: ok={successes} fail={failures}")

def parse_argument():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", "-d", default="prefix_cache_test.jsonl", help="JSONL input file")
    p.add_argument("--proxy", "-p", default="http://127.0.0.1:8000", help="proxy base URL (http://host:port)")
    p.add_argument("--use-requests", action="store_true", help="use requests instead of openai package")
    p.add_argument("--delay", type=float, default=0.05, help="delay between requests (s)")
    p.add_argument("--metric_url", default="http://10.0.13.1:8100", help="metrics from URL (http://host:port)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",help="model")
    p.add_argument("--max-tokens",type=int,default=100,help="response max tokens")
    p.add_argument("--strategy",default="RR",choices=['RR','Designate','Random'],help="request divert strategy")
    p.add_argument("--request-num",type=int,default=1,help="number of request")
    p.add_argument("--show-res",action="store_false",help="show response")
    p.add_argument("--prefill",default="10.0.13.1:8100",help="designated prefill node URL (host:port)")
    p.add_argument("--decode", default="10.0.14.1:8200", help="designated decode node URL (host:port)")
    args = p.parse_args()

    return args


def main():
    #parse argument
    args = parse_argument()

    #generate dataset file
    generate_input_file(args)

    #send to proxy server
    send_to_proxy_server(args)

    # fetch metrics and show
    show_metrics(args)

if __name__ == "__main__":
    main()
