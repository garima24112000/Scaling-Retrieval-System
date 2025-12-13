import argparse, json, time, datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_queries(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qfile", required=True)
    ap.add_argument("--results_dir", default="results_llm_only")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.results_dir) / f"{args.model.replace('/','_')}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Fix padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    queries = read_queries(args.qfile)
    per_query = []

    for q in queries:
        inputs = tokenizer(q, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=32)
        t1 = time.perf_counter()

        per_query.append((t1 - t0) * 1000.0)

    summary = {
        "model": args.model,
        "n_queries": len(queries),
        "avg_ms": sum(per_query)/len(per_query),
        "p50_ms": sorted(per_query)[len(per_query)//2],
        "p90_ms": sorted(per_query)[int(len(per_query)*0.9)],
        "raw_times_ms": per_query
    }

    with open(out_dir/"llm_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_dir)

if __name__ == "__main__":
    main()
