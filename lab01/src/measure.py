"""Steps 3-4: training time, single-sample latency, model size, peak memory, deployment fit."""
import csv
import os
import statistics
import time
import tracemalloc

import joblib
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone

from common import RESULTS, load_split, make_models, set_seeds

TRAIN_REPEATS = 5
INFER_REPEATS = 100
MB = 1024 ** 2

# Memory uses the lower bound of each tier: the model must fit the smallest device in it.
BUDGETS = {
    "Cloud":  {"mem_mb": 1024,       "latency_ms": 100, "size_mb": 500},
    "Edge":   {"mem_mb": 256,        "latency_ms": 50,  "size_mb": 50},
    "Mobile": {"mem_mb": 64,         "latency_ms": 20,  "size_mb": 10},
    "TinyML": {"mem_mb": 256 / 1024, "latency_ms": 10,  "size_mb": 100 / 1024},
}

PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROC.memory_info().rss / MB


def time_training(estimator, X, y) -> float:
    clone(estimator).fit(X, y)  # warm-up
    runs = []
    for _ in range(TRAIN_REPEATS):
        m = clone(estimator)
        t0 = time.perf_counter()
        m.fit(X, y)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


def time_inference(model, x) -> float:
    model.predict(x)  # warm-up
    runs = []
    for _ in range(INFER_REPEATS):
        t0 = time.perf_counter()
        model.predict(x)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs) * 1000


def peak_memory(fn, *args):
    """(process peak RSS MB, RSS growth over baseline MB, Python/NumPy heap peak KB)."""
    base = rss_mb()
    peak = memory_usage((fn, args), interval=0.001, max_usage=True)
    tracemalloc.start()
    fn(*args)
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, max(peak - base, 0.0), heap_peak / 1024


def fit_once(estimator, X, y):
    clone(estimator).fit(X, y)


def load_and_infer(path, x):
    model = joblib.load(path)
    for _ in range(INFER_REPEATS):
        model.predict(x)


def deployment_fit(row: dict) -> dict:
    mem_mb = row["infer_heap_peak_kb"] / 1024
    out = {"model": row["model"]}
    for tier, b in BUDGETS.items():
        checks = {
            "memory": mem_mb <= b["mem_mb"],
            "latency": row["infer_latency_ms"] <= b["latency_ms"],
            "size": row["size_kb"] / 1024 <= b["size_mb"],
        }
        failed = [k for k, ok in checks.items() if not ok]
        out[tier] = "yes" if not failed else "no (" + ", ".join(failed) + ")"
    return out


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    x_one = X_test[:1]
    rows, all_models = [], {}

    for name, estimator in make_models().items():
        train_s = time_training(estimator, X_train, y_train)
        train_peak, train_delta, train_heap = peak_memory(fit_once, estimator, X_train, y_train)

        model = clone(estimator).fit(X_train, y_train)
        all_models[name] = model
        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        latency_ms = time_inference(model, x_one)
        inf_peak, inf_delta, inf_heap = peak_memory(load_and_infer, path, x_one)

        row = {
            "model": name,
            "train_time_s": round(train_s, 4),
            "infer_latency_ms": round(latency_ms, 4),
            "size_bytes": size_b,
            "size_kb": round(size_b / 1024, 2),
            "train_peak_rss_mb": round(train_peak, 1),
            "train_rss_delta_mb": round(train_delta, 2),
            "train_heap_peak_kb": round(train_heap, 1),
            "infer_peak_rss_mb": round(inf_peak, 1),
            "infer_rss_delta_mb": round(inf_delta, 2),
            "infer_heap_peak_kb": round(inf_heap, 1),
        }
        rows.append(row)
        print(row)

    joblib.dump(all_models, RESULTS / "model.joblib")

    with open(RESULTS / "system_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fit_rows = [deployment_fit(r) for r in rows]
    with open(RESULTS / "deployment_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fit_rows[0]))
        w.writeheader()
        w.writerows(fit_rows)
    for r in fit_rows:
        print(r)


if __name__ == "__main__":
    main()
