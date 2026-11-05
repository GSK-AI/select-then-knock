"""PC-Knockoff comparison: single-split PC-Knockoff versus Headline and Secondary STK.

Run from the project root:

    # Software check only (reduced dimensions and STK budgets):
    python experiments/04-pc-knockoff-comparison/run.py --stage smoke

    # Evaluation (70 datasets): run k=0 and k>0 seed sets separately.
    python experiments/04-pc-knockoff-comparison/run.py --stage eval --supports 0 --seeds 10000-10029
    python experiments/04-pc-knockoff-comparison/run.py --stage eval --supports 1,3,10,20 --seeds 10000-10009

The runner sets one BLAS thread per process before importing numpy.

Each dataset is generated once in the parent, saved with its SHA-256, and given to
every arm. Each arm runs in its own spawn child, so peak RSS (getrusage high-water
mark) belongs to that arm alone. Arms:

- pc_knockoff: Algorithm 1 of Liu et al. (2022), see pc_knockoff.py. One shared
  evidence computation, thresholded at each alpha in --pc-alphas.
- stk_headline, stk_secondary: the headline and secondary presets, regression, no
  diagnostics. Split selections are observed without changing the estimator, and the
  recorded aggregate must equal the public get_support output.

Random streams per dataset seed s: data = s (_lib generators); PC partition =
SeedSequence([s, 1]); PC knockoff noise = SeedSequence([s, 2]); STK random_state = s.

Design defaults: n=800, p=5000, blocks of 20, rho=0.5, total SNR 0.5, beta from
_lib.beta_for_k, k=0 from _lib.make_null_dataset. No lineage term, so no
residualization. PC: n_screen=200, d=50, alpha in {0.1, 0.2, 0.5}.

--stage smoke: n=200, p=400, PC n_screen=60, d=20, STK n_splits=3 and reduced
budgets, plus a constant feature and a tied (rounded) noise feature. Implementation
evidence only.

A failed or timed-out arm is recorded as a failure, never as an empty selection.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from queue import Empty  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LIB_DIR = HERE.parent
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(HERE))
import _lib  # noqa: E402
import pc_knockoff as pk  # noqa: E402

from select_then_knock import SelectThenKnock  # noqa: E402

ARMS = ("pc_knockoff", "stk_headline", "stk_secondary")
STAGES = {
    "smoke": dict(n=200, p=400, n_screen=60, d=20, seeds="900,901", supports="0,3", stk_splits=3),
    "eval": dict(n=800, p=5000, n_screen=200, d=50, seeds=None, supports=None, stk_splits=25),
}
SMOKE_STK_BUDGET = dict(n_bootstrap=4, n_alphas=3, n_l1_ratios=2, cv_folds=2)


# ── Shared helpers ──────────────────────────────────────────────────────────
def sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value))


def support_metrics(selected, support):
    selected, support = set(map(int, selected)), set(map(int, support))
    tp = len(selected & support)
    fp = len(selected - support)
    return {
        "n_selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": len(support - selected),
        "fdp": fp / max(1, len(selected)),
        "power": tp / len(support) if support else None,
        "exact_recovery": int(selected == support),
        "firing": int(bool(selected)),
    }


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class TreeRSSSampler:
    """Peak summed RSS of this process and all descendants, sampled with ps.

    getrusage covers one process; STK's n_jobs workers are separate processes. ps
    reports RSS in KiB on Linux and macOS. Sampling can miss spikes shorter than the
    interval, so this is a lower bound on the true tree peak.
    """

    def __init__(self, interval_s=0.5):
        self.interval_s, self.peak_bytes, self.samples, self.max_processes = interval_s, 0, 0, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        proc = subprocess.Popen(["ps", "-A", "-o", "pid=,ppid=,rss="], stdout=subprocess.PIPE, text=True)
        rows, _ = proc.communicate()
        children, rss = {}, {}
        for line in rows.splitlines():
            parts = line.split()
            if len(parts) == 3 and int(parts[0]) != proc.pid:  # exclude ps itself
                pid, ppid, kib = map(int, parts)
                children.setdefault(ppid, []).append(pid)
                rss[pid] = kib
        stack, total, count = [os.getpid()], 0, 0
        while stack:
            pid = stack.pop()
            total, count = total + rss.get(pid, 0), count + 1
            stack.extend(children.get(pid, []))
        self.peak_bytes = max(self.peak_bytes, total * 1024)
        self.max_processes = max(self.max_processes, count)
        self.samples += 1

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()

    def record(self):
        return {
            "peak_tree_rss_bytes": self.peak_bytes,
            "tree_rss_samples": self.samples,
            "tree_rss_interval_s": self.interval_s,
            "max_tree_processes": self.max_processes,
        }


def stream_seed(seed, stream):
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def warning_summary(caught):
    counts = Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught)
    return [{"warning": key, "count": count} for key, count in counts.most_common()]


# ── Data ────────────────────────────────────────────────────────────────────
def make_dataset(cfg, k, seed):
    if k == 0:
        data = _lib.make_null_dataset(cfg["p"], cfg["rho"], seed, n=cfg["n"])
    else:
        data = _lib.generate_blocked_synthetic(
            n=cfg["n"],
            n_blocks=cfg["p"] // 20,
            block_size=20,
            rho=cfg["rho"],
            k_true=k,
            beta_true=_lib.beta_for_k(k),
            snr=cfg["snr"],
            seed=seed,
        )
    X, y, support = data["X"], data["y"], np.asarray(data["S_star"], dtype=int)
    modifications = []
    if cfg["stage"] == "smoke":
        # Implementation probes: one constant and one tied feature, both outside S*.
        free = [j for j in range(cfg["p"] - 1, -1, -1) if j not in set(support)]
        constant, tied = free[0], free[1]
        X = X.copy()
        X[:, constant] = 1.0
        X[:, tied] = np.round(X[:, tied])
        modifications = [{"constant_feature": constant}, {"rounded_tied_feature": tied}]
    return X, y, support, modifications


# ── Arm workers (spawn children) ────────────────────────────────────────────
class ObservedSTK(SelectThenKnock):
    """Record split outputs; the selector itself is unchanged."""

    def _one_split(self, X_arr, y_arr, cut, split_seed, per_run_fdr, knockoff_filter):
        seen = {}

        def observed_filter(Xc, yc, **kwargs):
            result = knockoff_filter(Xc, yc, **kwargs)
            seen.update(candidate_count=int(Xc.shape[1]), fallback_used=bool(result.fallback_used))
            return result

        t0 = time.perf_counter()
        selected = super()._one_split(X_arr, y_arr, cut, split_seed, per_run_fdr, observed_filter)
        self.recorded_splits_ = getattr(self, "recorded_splits_", [])
        self.recorded_splits_.append(
            {
                "split_seed": int(split_seed),
                "selected": np.asarray(selected, dtype=int).tolist(),
                "candidate_count": seen.get("candidate_count", 0),
                "fallback_used": seen.get("fallback_used"),
                "held_out_filter_called": bool(seen),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        return selected


def _child(arm, data_path, data_sha, cfg, seed, queue):
    started = time.perf_counter()
    try:
        with np.load(data_path) as data:
            X, y, support = data["X"], data["y"], data["S_star"]
        if sha256_arrays(X, y, support) != data_sha:
            raise RuntimeError("dataset hash mismatch in child")
        rss_after_load = peak_rss_bytes()
        with warnings.catch_warnings(record=True) as caught, TreeRSSSampler() as tree:
            warnings.simplefilter("always")
            if arm == "pc_knockoff":
                out = _run_pc(X, y, support, cfg, seed)
            else:
                out = _run_stk(arm.removeprefix("stk_"), X, y, support, cfg, seed)
        out.update(tree.record())
        out.update(
            ok=True,
            elapsed_s=time.perf_counter() - started,
            peak_rss_bytes=peak_rss_bytes(),
            peak_rss_after_load_bytes=rss_after_load,
            warnings=warning_summary(caught),
        )
    except Exception as error:
        out = {
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "peak_rss_bytes": peak_rss_bytes(),
        }
    queue.put(json.loads(json.dumps(out, default=_json_default)))


def _run_pc(X, y, support, cfg, seed):
    partition_seed, knockoff_seed = stream_seed(seed, 1), stream_seed(seed, 2)
    result = pk.pc_knockoff(
        X,
        y,
        n_screen=cfg["n_screen"],
        d=cfg["d"],
        alphas=cfg["pc_alphas"],
        partition_seed=partition_seed,
        knockoff_seed=knockoff_seed,
    )
    operating_points = {
        str(alpha): {
            "threshold": result.thresholds[alpha] if np.isfinite(result.thresholds[alpha]) else None,
            "selected": result.selected[alpha].tolist(),
            "metrics": support_metrics(result.selected[alpha], support),
        }
        for alpha in cfg["pc_alphas"]
    }
    return {
        "streams": {"partition_seed": partition_seed, "knockoff_seed": knockoff_seed},
        "n_screen_rows": len(result.screen_rows),
        "n_test_rows": len(result.test_rows),
        "rows_disjoint": not np.intersect1d(result.screen_rows, result.test_rows).size,
        "candidates": result.candidates.tolist(),
        "candidate_count": len(result.candidates),
        "support_screened": int(np.isin(support, result.candidates).sum()),
        "W": result.W.tolist(),
        "n_positive_W": int(np.sum(result.W > 0)),
        "knockoff_diagnostics": result.knockoff_diagnostics,
        "stage_elapsed_s": result.timings_s,
        "evidence_shared_across_alphas": True,
        "operating_points": operating_points,
    }


def _run_stk(preset, X, y, support, cfg, seed):
    overrides = {
        "random_state": seed,
        "run_diagnostics": False,
        "solver_backend": cfg["solver"],
        "n_jobs": cfg["stk_n_jobs"],
        "n_splits": cfg["stk_splits"],
    }
    if cfg["stage"] == "smoke":
        overrides.update(SMOKE_STK_BUDGET)
    model = ObservedSTK.from_preset(preset, task="regression", **overrides)
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0
    selected = np.asarray(model.get_support(indices=True, space="original"), dtype=int)
    splits = model.recorded_splits_
    counts = np.zeros(X.shape[1], dtype=int)
    for split in splits:
        counts[split["selected"]] += 1
    recomputed = np.flatnonzero(counts / len(splits) >= model.eta)
    if len(splits) != model.n_splits or not np.array_equal(recomputed, selected):
        raise AssertionError("recorded splits do not reproduce the public selection")
    out = {
        "parameters": model.get_params(),
        "stage_elapsed_s": {"fit": fit_s},
        "selected": selected.tolist(),
        "metrics": support_metrics(selected, support),
        "candidate_counts": [split["candidate_count"] for split in splits],
        "fallback_count": sum(split["fallback_used"] is True for split in splits),
        "splits_without_held_out_filter": sum(not split["held_out_filter_called"] for split in splits),
        "splits": splits,
        "pi_nonzero": {str(j): float(counts[j] / len(splits)) for j in np.flatnonzero(counts)},
    }
    return out


def run_child(arm, data_path, data_sha, cfg, seed):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(arm, str(data_path), data_sha, cfg, seed, queue))
    started = time.perf_counter()
    proc.start()
    out = None
    while out is None and time.perf_counter() - started < cfg["timeout_s"]:
        try:
            out = queue.get(timeout=2)
        except Empty:
            if not proc.is_alive():
                out = {"ok": False, "error": f"child exited with code {proc.exitcode}, no result"}
    if out is None:
        out = {"ok": False, "timed_out": True, "error": f"timeout after {cfg['timeout_s']} s"}
    proc.join(timeout=30)
    if proc.is_alive():
        proc.kill()
        proc.join()
    out.setdefault("timed_out", False)
    out["wall_s"] = time.perf_counter() - started
    out["exitcode"] = proc.exitcode
    return out


# ── Provenance and orchestration ────────────────────────────────────────────
def provenance(cfg):
    import scipy
    import sklearn

    import select_then_knock

    source = Path(select_then_knock.__file__).resolve().parent
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}
    if cfg["solver"] == "celer":
        packages["celer"] = importlib.metadata.version("celer")
    upstream = HERE / "outputs" / "upstream" / "pc_screen" / "PC_screen.py"
    return {
        "stk_source_path": str(source),
        "stk_source_sha256": {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))},
        "runner_sha256": file_hash(__file__),
        "pc_knockoff_sha256": file_hash(HERE / "pc_knockoff.py"),
        "lib_sha256": file_hash(LIB_DIR / "_lib.py"),
        "upstream_pc_screen": {
            "revision": pk.UPSTREAM_REVISION,
            "expected_sha256": pk.UPSTREAM_PC_SCREEN_SHA256,
            "local_sha256": file_hash(upstream) if upstream.exists() else None,
        },
        "module_paths": {m.__name__: m.__file__ for m in (np, scipy, sklearn, select_then_knock)},
        "packages": packages,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy_build": np.show_config(mode="dicts"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "hostname": socket.gethostname(),
        "slurm": {
            name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")
        },
        "worktree": worktree_state(),
    }


def worktree_state():
    """Record local fingerprints without requiring an installed distribution or git."""
    return {
        "source": "local-source",
        "commit": _lib.git_sha(),
        "status": _lib.git_status(),
        "source_sha256": _lib.source_fingerprints(),
    }


def run_dataset(cfg, k, seed, source):
    X, y, support, modifications = make_dataset(cfg, k, seed)
    data_sha = sha256_arrays(X, y, support)
    data_path = cfg["out"] / "data" / f"k-{k}_seed-{seed}.npz"
    if data_path.exists():
        with np.load(data_path) as old:
            if sha256_arrays(old["X"], old["y"], old["S_star"]) != data_sha:
                raise ValueError(f"Regenerated dataset differs from saved array: {data_path}")
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data_path, X=X, y=y, S_star=support)
    for arm in cfg["arms"]:
        path = cfg["out"] / f"k-{k}" / f"seed-{seed}" / f"{arm}.json"
        signature = json.loads(
            json.dumps(
                {
                    "arm": arm,
                    "stage": cfg["stage"],
                    "k": k,
                    "dataset_seed": seed,
                    "dataset_sha256": data_sha,
                    "modifications": modifications,
                    "config": {key: value for key, value in cfg.items() if key not in ("out", "arms")},
                    "runner_sha256": source["runner_sha256"],
                    "pc_knockoff_sha256": source["pc_knockoff_sha256"],
                    "stk_source_sha256": source["stk_source_sha256"],
                    "lib_sha256": source["lib_sha256"],
                    "packages": source["packages"],
                    "python": source["python"],
                },
                default=_json_default,
            )
        )
        if path.exists():
            existing = json.loads(path.read_text())
            if existing["signature"] != signature:
                raise ValueError(f"Refusing to reuse mismatched result: {path}")
            if existing.get("complete"):
                print(f"SKIP {path.relative_to(cfg['out'])}", flush=True)
                continue
        print(f"START k={k} seed={seed} arm={arm}", flush=True)
        result = run_child(arm, data_path, data_sha, cfg, seed)
        record = {
            "signature": signature,
            "provenance": source,
            "S_star": support.tolist(),
            "complete": bool(result.get("ok")),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        write_json(path, record)
        if result.get("ok"):
            if arm == "pc_knockoff":
                brief = {a: v["metrics"] for a, v in result["operating_points"].items()}
            else:
                brief = result["metrics"]
            print(
                f"DONE  k={k} seed={seed} arm={arm} wall={result['wall_s']:.1f}s "
                f"rss={result['peak_rss_bytes'] / 2**20:.0f}MiB {brief}",
                flush=True,
            )
        else:
            print(f"FAIL  k={k} seed={seed} arm={arm}: {result.get('error')}", flush=True)


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            low, high = map(int, part.split("-"))
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--seeds", help="comma list and/or ranges, e.g. 10000-10009")
    parser.add_argument("--supports", help="comma list of k")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--pc-alphas", default="0.1,0.2,0.5")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--solver", choices=("sklearn", "celer"), default="celer")
    parser.add_argument("--stk-n-jobs", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=4 * 3600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stage = STAGES[args.stage]
    seeds = args.seeds or stage["seeds"]
    supports = args.supports or stage["supports"]
    if seeds is None or supports is None:
        parser.error("--stage eval needs explicit --seeds and --supports")
    arms = args.arms.split(",")
    if not set(arms) <= set(ARMS):
        parser.error(f"arms must be in {ARMS}")
    cfg = {
        "stage": args.stage,
        "n": stage["n"],
        "p": stage["p"],
        "rho": args.rho,
        "snr": args.snr,
        "n_screen": stage["n_screen"],
        "d": stage["d"],
        "pc_alphas": [float(a) for a in args.pc_alphas.split(",")],
        "stk_splits": stage["stk_splits"],
        "stk_n_jobs": args.stk_n_jobs,
        "solver": args.solver,
        "timeout_s": args.timeout_s,
        "arms": arms,
        "out": args.out or HERE / "outputs" / args.stage,
    }
    return cfg, parse_seeds(seeds), [int(k) for k in supports.split(",")]


def main():
    cfg, seeds, supports = parse_args()
    source = provenance(cfg)
    cfg["out"].mkdir(parents=True, exist_ok=True)
    tag = f"{socket.gethostname()}-{os.getpid()}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    write_json(cfg["out"] / "provenance" / f"{tag}.json", source)
    print(f"PC-Knockoff vs STK: stage={cfg['stage']} k={supports} seeds={seeds} arms={cfg['arms']}", flush=True)
    for k in supports:
        for seed in seeds:
            run_dataset(cfg, k, seed, source)


if __name__ == "__main__":
    main()
