"""
Classify every ethical query into one of 13 product domains with a local
Qwen 2.5-3B model (MLX, Apple Silicon). Each worker process loads its own copy
of the model; three workers fit in 16 GB of unified memory.

    python src/send_query_llm.py [--workers 3] [--resume]

Output: data/ethics_classified_5pct.csv
"""

import os, sys, time, queue, argparse, multiprocessing as mp
from datetime import datetime
import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
INPUT_CSV  = os.path.join(DATA_DIR, "ethical_logical_5pct.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "ethics_classified_5pct.csv")
CHECKPOINT = os.path.join(DATA_DIR, "ethics_classified_5pct.checkpoint.csv")
LOG_FILE   = os.path.join(DATA_DIR, "llm_run.log")

MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"

ETHICS_CATEGORIES = [
    "environmental_sustainability", "fair_trade_labor_human_rights",
    "animal_welfare", "privacy_data_ethics", "accessibility_inclusion",
]

VALID_DOMAINS = [
    "Books", "Films TV Music and Games", "Electronics and Computers",
    "Home Garden and DIY", "Toys Children and Baby",
    "Clothes Shoes and Watches", "Sports and Outdoors",
    "Food and Grocery", "Health and Beauty", "Car and Motorbike",
    "Business Industry and Science", "Stationery and Office Supplies",
    "None of the above",
]
VALID_SET = set(VALID_DOMAINS)

SYSTEM_PROMPT = (
    "You classify Amazon search queries into exactly one product domain.\n"
    "Reply with ONLY the domain name. No explanation.\n\n"
    "Domains:\n" + "\n".join(f"- {d}" for d in VALID_DOMAINS) + "\n\n"
    "Examples:\n"
    "organic green tea → Food and Grocery\n"
    "reusable grocery bags → Home Garden and DIY\n"
    "organic tampons → Health and Beauty\n"
    "privacy screen protector laptop → Electronics and Computers\n"
    "eco friendly dog treats → Home Garden and DIY\n"
    "fair trade chocolate → Food and Grocery\n"
    "organic baby onesie → Clothes Shoes and Watches\n"
    "artisan sourdough book → Books\n"
    "biodegradable phone case → Electronics and Computers\n"
    "natural cat litter → Home Garden and DIY\n"
)

# Common model paraphrases mapped back to a domain
ALIASES = {
    "cosmetics": "Health and Beauty", "beauty": "Health and Beauty",
    "makeup": "Health and Beauty", "personal care": "Health and Beauty",
    "skincare": "Health and Beauty", "fragrance": "Health and Beauty",
    "cosmetics and beauty": "Health and Beauty",
    "toiletries": "Health and Beauty",
    "toilets and washrooms": "Home Garden and DIY",
    "bathroom": "Home Garden and DIY",
    "clothing": "Clothes Shoes and Watches",
    "fashion": "Clothes Shoes and Watches",
    "apparel": "Clothes Shoes and Watches",
    "shoes": "Clothes Shoes and Watches",
    "jewelry": "Clothes Shoes and Watches",
    "accessories": "Clothes Shoes and Watches",
    "kitchen": "Home Garden and DIY", "furniture": "Home Garden and DIY",
    "garden": "Home Garden and DIY", "home": "Home Garden and DIY",
    "tools": "Home Garden and DIY", "hardware": "Home Garden and DIY",
    "home and kitchen": "Home Garden and DIY",
    "household": "Home Garden and DIY",
    "pet": "Home Garden and DIY", "pets": "Home Garden and DIY",
    "pet supplies": "Home Garden and DIY",
    "pets and home": "Home Garden and DIY",
    "pets and animals": "Home Garden and DIY",
    "cooking and recipes": "Books", "cookbooks": "Books",
    "foods and grocery": "Food and Grocery",
    "grocery": "Food and Grocery", "food": "Food and Grocery",
    "beverages": "Food and Grocery", "snacks": "Food and Grocery",
    "computer": "Electronics and Computers",
    "laptop": "Electronics and Computers",
    "phone": "Electronics and Computers",
    "tablet": "Electronics and Computers",
    "camera": "Electronics and Computers",
    "audio": "Electronics and Computers",
    "tv": "Films TV Music and Games", "music": "Films TV Music and Games",
    "games": "Films TV Music and Games", "gaming": "Films TV Music and Games",
    "movies": "Films TV Music and Games",
    "toys": "Toys Children and Baby", "baby": "Toys Children and Baby",
    "kids": "Toys Children and Baby", "children": "Toys Children and Baby",
    "sports": "Sports and Outdoors", "fitness": "Sports and Outdoors",
    "outdoor": "Sports and Outdoors", "camping": "Sports and Outdoors",
    "exercise": "Sports and Outdoors",
    "automotive": "Car and Motorbike", "car": "Car and Motorbike",
    "vehicle": "Car and Motorbike",
    "office": "Stationery and Office Supplies",
    "stationery": "Stationery and Office Supplies",
    "industrial": "Business Industry and Science",
    "science": "Business Industry and Science",
}



class Tee:
    """Write to the terminal and, timestamped, to the log file."""
    def __init__(self, log_path, stream):
        self._stream = stream
        self._file = open(log_path, "a", encoding="utf-8")

    def write(self, msg):
        self._stream.write(msg)
        self._stream.flush()
        if msg.strip():
            ts = datetime.now().strftime("%H:%M:%S")
            self._file.write(f"{ts}  {msg}")
        else:
            self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def match_domain(raw):
    """Map a model response to a domain name, or None."""
    s = raw.strip().rstrip(".")
    if s in VALID_SET:
        return s
    sl = s.lower()
    for d in VALID_DOMAINS:
        if d.lower() == sl:
            return d
    for d in VALID_DOMAINS:
        if d.lower() in sl or sl in d.lower():
            return d
    return ALIASES.get(sl)


def save(queries_df, done, path):
    keep = (["query_id", "session_id", "final_search_term", "popularity"]
            + [f"{c}_prefix" for c in ETHICS_CATEGORIES]
            + [f"{c}_final"  for c in ETHICS_CATEGORIES])
    keep = [c for c in keep if c in queries_df.columns]
    out = queries_df[keep].copy()
    out["product_domain"] = queries_df["query_id"].map(done).values
    out.to_csv(path, index=False)



def worker_fn(worker_id, tasks, result_queue):
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_NAME)

    for global_idx, query_id, query in tasks:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": query}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate(model, tokenizer, prompt=prompt,
                       max_tokens=15, verbose=False).strip()
        domain = match_domain(raw)
        status = "" if domain else f" [unmatched: {raw!r}]"
        print(f"  W{worker_id} [{global_idx:,}] {query!r:50s} -> {domain}{status}",
              flush=True)
        result_queue.put((query_id, domain))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default=INPUT_CSV)
    ap.add_argument("--output",  default=OUTPUT_CSV)
    ap.add_argument("--resume",  action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    sys.stdout = Tee(LOG_FILE, sys.__stdout__)
    sys.stderr = Tee(LOG_FILE, sys.__stderr__)

    print(f"Workers : {args.workers}")
    print(f"Log     : {LOG_FILE}")

    df = pd.read_csv(args.input)
    ethics_final = [f"{c}_final" for c in ETHICS_CATEGORIES]
    df = df[df[ethics_final].any(axis=1)].reset_index(drop=True)
    queries_df = df.drop_duplicates(subset="query_id").reset_index(drop=True)
    total = len(queries_df)
    print(f"Queries to classify: {total:,}")

    # The checkpoint lists every query; only those with a label are done
    done = {}
    if args.resume and os.path.exists(CHECKPOINT):
        ckpt = pd.read_csv(CHECKPOINT).dropna(subset=["product_domain"])
        done = dict(zip(ckpt["query_id"], ckpt["product_domain"]))
        print(f"Resuming from checkpoint: {len(done):,} already done")
    start = len(done)

    remaining = queries_df[~queries_df["query_id"].isin(done)]
    n = len(remaining)
    print(f"Remaining: {n:,}\n")
    if n == 0:
        return

    tasks = [(start + i + 1, row.query_id,
              row.final_search_term if pd.notna(row.final_search_term) else "")
             for i, row in enumerate(remaining.itertuples())]
    worker_tasks = [tasks[w::args.workers] for w in range(args.workers)]

    result_queue = mp.Queue()
    processes = [mp.Process(target=worker_fn, args=(w, worker_tasks[w], result_queue))
                 for w in range(args.workers)]
    t0 = time.time()
    for p in processes:
        p.start()

    # Collect results, checkpointing every 500
    collected = 0
    while collected < n:
        try:
            query_id, domain = result_queue.get(timeout=300)
            done[query_id] = domain
            collected += 1

            if collected % 500 == 0:
                elapsed = time.time() - t0
                rate = collected / elapsed
                eta = (n - collected) / rate
                print(f"\n  {collected + start:,}/{total:,}  "
                      f"{elapsed:.0f}s  {rate:.1f} q/s  "
                      f"ETA {eta/3600:.1f}h\n", flush=True)
                save(queries_df, done, CHECKPOINT)
        except queue.Empty:
            if not any(p.is_alive() for p in processes):
                print(f"All workers exited after {collected} results.")
                break

    for p in processes:
        p.join()

    save(queries_df, done, args.output)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    elapsed = time.time() - t0
    print(f"\nSaved {args.output}")
    print(f"Total time: {elapsed/3600:.1f} h  ({collected/elapsed:.1f} q/s)")

    dist = pd.read_csv(args.output)["product_domain"].value_counts(dropna=False)
    print("\nDomain breakdown:")
    for d, c in dist.items():
        print(f"  {c:>6,}  {d}")


if __name__ == "__main__":
    main()
