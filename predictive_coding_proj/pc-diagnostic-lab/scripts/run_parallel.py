#!/usr/bin/env python3
"""Fan out independent jobs and merge them into one metrics.json.

C2 jobs are (cell, seed). C7 jobs are (unroll_k, lambda_slow, seed).
C1, C3 and C8 jobs are one seed each. Each job is a subprocess so one hung
settle does not kill the grid. The merged file is what a single sequential
multi-seed run would have written (tests/test_parallel.py checks this).

    python scripts/run_parallel.py c1 --workers 3 --device cpu --seeds 0 1 2
    python scripts/run_parallel.py c2 --workers 8 --device cpu --seeds 0 1 2
    python scripts/run_parallel.py c7 --workers 8 --device cpu --seeds 0 1 2

Each job is single-threaded Python settle, so throughput follows the number of
fast cores, not the GPU. Measured on an M5 (4 performance cores): 4 workers
≈ 3.5x one worker, 8 workers ≈ 4.4x.

Continue a run that lost jobs (finished shards are skipped):

    python scripts/run_parallel.py c2 --resume results/c2_mitigation_grid/STAMP --seeds 0 1 2

One Slurm array task per job, all writing into one shared run dir:

    python scripts/run_parallel.py c2 --warmup-only --device cpu
    python scripts/run_parallel.py c2 --resume $RUN --shard $SLURM_ARRAY_TASK_ID/27 --workers 1 --device cpu
    python scripts/run_parallel.py c2 --merge-only $RUN

The merge refuses an incomplete grid unless --allow-partial is given.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import c1_rollout_collapse as c1_exp
from experiments import c2_mitigation_grid as c2_exp
from experiments import c3_copy_detection as c3_exp
from experiments import c7_slowness_sweep as c7_exp
from experiments import c8_amortized_contrast as c8_exp
from src.config import resolve_config
from src.utils import finish_run, get_device, link_latest, mean_std, write_yaml

C2_CELLS = [c["name"] for c in c2_exp.CELLS]
EXP_NAME = {
    "c1": "c1_rollout_collapse",
    "c2": "c2_mitigation_grid",
    "c3": "c3_copy_detection",
    "c7": "c7_slowness_sweep",
    "c8": "c8_amortized_contrast",
}
# One job per seed; each module exposes aggregate(per_seed_records, cfg).
SEED_CLAIMS = {"c1": c1_exp, "c3": c3_exp, "c8": c8_exp}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Parallel C1 / C2 / C3 / C7 / C8 runner")
    p.add_argument("claim", choices=sorted(EXP_NAME))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overlay", nargs="+", default=None)
    p.add_argument("--cells", nargs="+", default=None, help="C2: subset of cell names")
    p.add_argument("--lambda-slow", nargs="+", type=float, default=None, help="C7: subset of λ")
    p.add_argument("--unroll-k", nargs="+", type=int, default=None, help="C7: subset of unroll_k")
    p.add_argument(
        "--resume",
        default=None,
        help="run into this dir instead of a new stamp; shards that already have metrics.json are skipped",
    )
    p.add_argument(
        "--shard",
        default=None,
        help="I/N: run only jobs[I::N] (0-based) and do not merge. Needs --resume. For Slurm arrays.",
    )
    p.add_argument("--warmup-only", action="store_true", help="pretrain/cache dictionaries, then exit")
    p.add_argument("--count-jobs", action="store_true", help="print the number of jobs in the grid, then exit")
    p.add_argument(
        "--merge-only",
        default=None,
        help="merge shards under this run dir (must contain shards/*/metrics.json)",
    )
    p.add_argument("--allow-partial", action="store_true", help="merge even if expected shards are missing")
    p.add_argument("--no-latest", action="store_true", help="do not retarget results/<exp>/latest")
    return p.parse_args(argv)


def available_cpus():
    """Cores this process may actually use. Under Slurm os.cpu_count() reports
    the whole node, not the allocation, which would start 64 BLAS threads on a
    1-core job."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var, "")
        if val.isdigit() and int(val) > 0:
            return int(val)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def default_workers(device):
    name = str(get_device(device or "auto"))
    if name in {"cuda", "mps"}:
        return 2
    return max(1, min(8, available_cpus()))


def resolve_overlays(overlays):
    out = []
    for extra in overlays or []:
        path = Path(extra)
        if not path.is_absolute():
            cand = ROOT / path
            path = cand if cand.exists() else path.resolve()
        out.append(str(path))
    return out


def c2_jobs(cfg, seeds, cells):
    names = cells or cfg.get("c2", {}).get("cells") or C2_CELLS
    unknown = [n for n in names if n not in C2_CELLS]
    if unknown:
        raise SystemExit(f"unknown C2 cells {unknown}; known: {C2_CELLS}")
    return [{"cell": name, "seed": seed} for name in names for seed in seeds]


def c7_jobs(cfg, seeds, lambdas=None, ks=None):
    lams = lambdas if lambdas is not None else cfg.get("c7", {}).get("lambda_slow", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    unroll = ks if ks is not None else cfg.get("c7", {}).get("unroll_k", [0, 5])
    return [{"unroll_k": k, "lambda_slow": lam, "seed": seed} for k in unroll for lam in lams for seed in seeds]


def seed_jobs(seeds):
    return [{"seed": seed} for seed in seeds]


def make_jobs(claim, cfg, seeds, cells=None, lambdas=None, ks=None):
    if claim in SEED_CLAIMS:
        return seed_jobs(seeds)
    if claim == "c2":
        return c2_jobs(cfg, seeds, cells)
    return c7_jobs(cfg, seeds, lambdas=lambdas, ks=ks)


def job_tag(claim, job):
    if claim in SEED_CLAIMS:
        return f"s{job['seed']}"
    if claim == "c2":
        return f"{job['cell']}_s{job['seed']}"
    return f"k{job['unroll_k']}_l{job['lambda_slow']}_s{job['seed']}"


def write_job_overlay(job_dir, claim, job):
    if claim in SEED_CLAIMS:
        payload = {}  # the seed is passed with --seeds; nothing else varies
    elif claim == "c2":
        payload = {"c2": {"cells": [job["cell"]]}}
    else:
        payload = {"c7": {"lambda_slow": [job["lambda_slow"]], "unroll_k": [job["unroll_k"]]}}
    tag = job_tag(claim, job)
    path = Path(job_dir) / f"{tag}.yaml"
    write_yaml(path, payload)
    return path, tag


def job_base_cfg(cfg, claim, n_train=None, epochs=None):
    """The config a job subprocess will see, as far as the dictionary cache key
    (src.spatial_pc.dictionary_key) is concerned. The key hashes cfg['spatial']
    and data.n_train, so the warmup has to apply the same edits the job does:
    --n-train, and C7's spatial.freeze=False. Otherwise every job misses the
    cache and they all pretrain the same dictionary at once."""
    out = copy.deepcopy(cfg)
    if n_train is not None:
        out["data"]["n_train"] = n_train
    if epochs is not None:
        out["temporal"]["epochs"] = epochs
    if claim == "c7":
        out["spatial"]["freeze"] = False
    return out


def warmup_dictionaries(cfg, seeds, device):
    """Pretrain once per seed so parallel jobs do not race on dictionary.pt.
    `cfg` must already be job_base_cfg(...) so the cache key matches the jobs'."""
    from src.experiment import ensure_dictionary, load_data
    from src.utils import seed_everything

    torch_dev = get_device(device or cfg.get("device") or "auto")
    print(f"warming dictionaries for seeds={list(seeds)} on {torch_dev}", flush=True)
    for seed in seeds:
        seed_everything(seed)
        cell_cfg = copy.deepcopy(cfg)
        cell_cfg["seed"] = seed
        train, _, _, _ = load_data(cell_cfg, ROOT, seed)
        ensure_dictionary(cell_cfg, train, torch_dev, ROOT, seed=seed)


def run_job(claim, job, device, smoke, overlays, n_train, epochs, shard_root, threads=None):
    shard_root = Path(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    overlay, tag = write_job_overlay(shard_root, claim, job)
    run_dir = shard_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    script = ROOT / "experiments" / f"{EXP_NAME[claim]}.py"
    cmd = [sys.executable, str(script), "--seeds", str(job["seed"])]
    if device:
        cmd += ["--device", device]
    if smoke:
        cmd.append("--smoke")
    if n_train is not None:
        cmd += ["--n-train", str(n_train)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    all_overlays = list(overlays or []) + [str(overlay)]
    cmd += ["--overlay", *all_overlays]
    env = os.environ.copy()
    env["PC_LAB_RUN_DIR"] = str(run_dir)
    env["PC_LAB_SKIP_LATEST"] = "1"
    if threads:
        # W processes x (all cores) BLAS threads each oversubscribes badly on a
        # many-core node. Respect an explicit setting from the caller.
        env.setdefault("OMP_NUM_THREADS", str(threads))
        env.setdefault("MKL_NUM_THREADS", str(threads))
    print(">>>", " ".join(cmd), flush=True)
    with open(run_dir / "log.txt", "w") as log:
        subprocess.check_call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return run_dir


def _check_seeds(label, seeds):
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        raise SystemExit(f"{label}: seed(s) {dup} appear in more than one shard — refusing to double-count")


def merge_c2(shard_dirs):
    """Merge per-(cell, seed) shards. Values keep the order of `shard_dirs`."""
    by_name = {}
    headline = 10
    for d in shard_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        headline = m.get("headline_split", headline)
        for rec in m.get("cells") or []:
            by_name.setdefault(rec["name"], []).append(rec)
    drop = {
        "long_mse_mean", "long_mse_std", "per_seed", "seeds",
        "copy_last_long_mse_mean", "copy_last_long_mse_std",
        "mean_frame_long_mse_mean", "mean_frame_long_mse_std",
    }
    cells = []
    for name, recs in by_name.items():
        base = {k: v for k, v in recs[0].items() if k not in drop}
        per_seed, seeds, cl, mf = [], [], [], []
        for r in recs:
            per_seed.extend(r.get("per_seed") or [])
            seeds.extend(r.get("seeds") or [])
            if r.get("copy_last_long_mse_mean") is not None:
                cl.append(r["copy_last_long_mse_mean"])
            if r.get("mean_frame_long_mse_mean") is not None:
                mf.append(r["mean_frame_long_mse_mean"])
        _check_seeds(f"C2 cell {name}", seeds)
        mu, sd = mean_std(per_seed)
        cl_mu, cl_sd = mean_std(cl) if cl else (float("nan"), 0.0)
        mf_mu, mf_sd = mean_std(mf) if mf else (float("nan"), 0.0)
        cells.append({
            **base,
            "name": name,
            "long_mse_mean": mu,
            "long_mse_std": sd,
            "copy_last_long_mse_mean": cl_mu,
            "copy_last_long_mse_std": cl_sd,
            "mean_frame_long_mse_mean": mf_mu,
            "mean_frame_long_mse_std": mf_sd,
            "per_seed": per_seed,
            "seeds": seeds,
        })
    order = {n: i for i, n in enumerate(C2_CELLS)}
    cells.sort(key=lambda c: order.get(c["name"], 99))
    seed_sets = {tuple(sorted(c["seeds"])) for c in cells}
    metrics = c2_exp.summarize(cells, headline)
    metrics["seeds_consistent_across_cells"] = len(seed_sets) <= 1
    return metrics


def merge_c7(shard_dirs):
    """Merge per-(unroll_k, lambda, seed) shards. Values keep the order of `shard_dirs`."""
    by_key = {}
    for d in shard_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        for rec in m.get("sweep") or []:
            key = (rec["unroll_k"], rec["lambda_slow"])
            by_key.setdefault(key, []).append(rec)

    def _cat(recs, path):
        out = []
        for r in recs:
            cur = r.get("per_seed") or {}
            out.extend(cur.get(path) or [])
        return out

    sweep = []
    for (unroll_k, lam), recs in sorted(by_key.items()):
        cons_np = _cat(recs, "cons_cos_no_pull")
        frac_np = _cat(recs, "frac_rel_no_pull")
        cons_wp = _cat(recs, "cons_cos_with_pull")
        frac_wp = _cat(recs, "frac_rel_with_pull")
        drift = _cat(recs, "dict_drift")
        val = _cat(recs, "val_mse")
        cl = _cat(recs, "copy_last_mse")
        grad = _cat(recs, "grad_path")
        seeds = _cat(recs, "seeds")
        _check_seeds(f"C7 cell k={unroll_k} λ={lam}", seeds)
        mu_np, sd_np = mean_std(cons_np)
        fmu_np, fsd_np = mean_std(frac_np)
        mu_wp, sd_wp = mean_std(cons_wp)
        fmu_wp, fsd_wp = mean_std(frac_wp)
        drift_mu, _ = mean_std(drift) if drift else (float("nan"), 0.0)
        val_mu, val_sd = mean_std(val) if val else (float("nan"), 0.0)
        cl_mu, cl_sd = mean_std(cl) if cl else (float("nan"), 0.0)
        sweep.append({
            "lambda_slow": lam,
            "unroll_k": unroll_k,
            "cons_cos_no_pull": mu_np,
            "cons_cos_no_pull_std": sd_np,
            "frac_rel_no_pull": fmu_np,
            "frac_rel_no_pull_std": fsd_np,
            "cons_cos_with_pull": mu_wp,
            "cons_cos_with_pull_std": sd_wp,
            "frac_rel_with_pull": fmu_wp,
            "frac_rel_with_pull_std": fsd_wp,
            "dict_drift": drift_mu,
            "val_mse": val_mu,
            "val_mse_std": val_sd,
            "copy_last_mse": cl_mu,
            "copy_last_mse_std": cl_sd,
            "grad_path": bool(any(grad)),
            "per_seed": {
                "seeds": seeds,
                "cons_cos_no_pull": cons_np,
                "frac_rel_no_pull": frac_np,
                "cons_cos_with_pull": cons_wp,
                "frac_rel_with_pull": frac_wp,
                "dict_drift": drift,
                "val_mse": val,
                "copy_last_mse": cl,
                "grad_path": grad,
            },
            "cons_cos_mean": mu_np,
            "cons_cos_std": sd_np,
            "frac_rel_mean": fmu_np,
            "frac_rel_std": fsd_np,
        })
    # Same order the sequential script emits: unroll_k outer, lambda inner.
    lams = sorted({s["lambda_slow"] for s in sweep})
    metrics = c7_exp.summarize(sweep, lams)
    seed_sets = {tuple(sorted(s["per_seed"]["seeds"])) for s in sweep}
    metrics["seeds_consistent_across_cells"] = len(seed_sets) <= 1
    return metrics


def merge_seeds(claim, shard_dirs, cfg):
    """Merge one-seed shards of C1 / C3 / C8: concatenate their per-seed records
    in the order of `shard_dirs` and re-run the experiment's own aggregate()."""
    records = []
    for d in shard_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        records.extend(m.get("seeds") or [])
    _check_seeds(claim.upper(), [r["seed"] for r in records])
    return SEED_CLAIMS[claim].aggregate(records, cfg)


def link_seed_dirs(parent, shard_dirs):
    """A sequential C1/C3 run keeps checkpoints and panels in <run>/seed<k>/.
    Point the same names at the shard copies so both layouts look alike."""
    for d in shard_dirs:
        for sub in sorted(Path(d).glob("seed*")):
            link = Path(parent) / sub.name
            if sub.is_dir() and not link.exists() and not link.is_symlink():
                try:
                    link.symlink_to(Path("shards") / Path(d).name / sub.name, target_is_directory=True)
                except OSError:
                    pass


def collect_shards(shard_root):
    shard_root = Path(shard_root)
    if not shard_root.is_dir():
        return []
    return sorted(p for p in shard_root.iterdir() if p.is_dir() and (p / "metrics.json").exists())


def shard_status(claim, jobs, shard_root):
    """Split expected jobs into (done_dirs in job order, missing tags)."""
    done, missing = [], []
    for job in jobs:
        d = Path(shard_root) / job_tag(claim, job)
        if (d / "metrics.json").exists():
            done.append(d)
        else:
            missing.append(job_tag(claim, job))
    return done, missing


def write_merged(claim, parent, jobs, cfg, allow_partial=False, link=True, extra=None):
    """Merge the shards of `jobs` under parent/shards. Refuses a partial grid
    unless allow_partial, so a mean over 2 of 3 seeds is never written silently."""
    parent = Path(parent)
    done, missing = shard_status(claim, jobs, parent / "shards")
    if not done:
        raise SystemExit(f"no shard metrics under {parent / 'shards'}")
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)}/{len(jobs)} expected shards are missing: {missing[:6]}"
            f"{' ...' if len(missing) > 6 else ''}\n"
            f"finish them with --resume {parent}, or pass --allow-partial to merge anyway. "
            f"If this run used a subset or --smoke, pass the same --cells/--seeds/--smoke flags to the merge."
        )
    if claim in SEED_CLAIMS:
        metrics = merge_seeds(claim, done, cfg)
        link_seed_dirs(parent, done)
    else:
        metrics = merge_c2(done) if claim == "c2" else merge_c7(done)
    metrics["n_jobs_expected"] = len(jobs)
    metrics["n_jobs_merged"] = len(done)
    metrics["missing_shards"] = missing
    if missing:
        metrics["summary"] += f"  | PARTIAL: {len(done)}/{len(jobs)} jobs"
    out_cfg = copy.deepcopy(cfg)
    out_cfg["parallel"] = {"claim": claim, "jobs": jobs, **(extra or {})}
    finish_run(parent, out_cfg, metrics, root=ROOT, summary=metrics["summary"])
    if link:
        link_latest(ROOT, EXP_NAME[claim], parent)
    return metrics


def parse_shard(spec, n_jobs):
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard wants I/N, got {spec!r}")
    if not (0 <= i < n):
        raise SystemExit(f"--shard {spec}: need 0 <= I < N")
    return i, n


def main(argv=None):
    args = parse_args(argv)
    claim = args.claim
    overlays = resolve_overlays(args.overlay)
    cfg = resolve_config(overlays=overlays, smoke=args.smoke, root=ROOT)
    cfg = job_base_cfg(cfg, claim, n_train=args.n_train, epochs=args.epochs)
    seeds = args.seeds if args.seeds is not None else list(range(cfg.get("n_seeds", 3)))
    device = args.device or cfg.get("device") or "auto"
    workers = args.workers if args.workers is not None else default_workers(device)
    jobs = make_jobs(claim, cfg, seeds, cells=args.cells, lambdas=args.lambda_slow, ks=args.unroll_k)
    extra = {"seeds": list(seeds), "workers": workers, "smoke": bool(args.smoke), "overlays": overlays}

    if args.count_jobs:
        print(len(jobs))
        return

    if args.merge_only:
        write_merged(claim, args.merge_only, jobs, cfg, args.allow_partial, link=not args.no_latest, extra=extra)
        return

    if args.warmup_only:
        warmup_dictionaries(cfg, seeds, device)
        return

    if args.shard and not args.resume:
        raise SystemExit("--shard needs --resume DIR so every array task writes into the same run dir")
    if args.resume:
        parent = Path(args.resume).resolve()
    else:
        parent = ROOT / "results" / EXP_NAME[claim] / datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_root = parent / "shards"

    todo = jobs
    if args.shard:
        i, n = parse_shard(args.shard, len(jobs))
        todo = jobs[i::n]
    _, missing = shard_status(claim, todo, shard_root)
    pending = [j for j in todo if job_tag(claim, j) in set(missing)]
    print(
        f"{claim}: {len(jobs)} jobs in grid, {len(todo)} assigned here, "
        f"{len(todo) - len(pending)} already done, {len(pending)} to run, "
        f"{workers} workers, device={device}\nrun dir: {parent}",
        flush=True,
    )

    errors = []
    if pending:
        warmup_dictionaries(cfg, sorted({j["seed"] for j in pending}), device)
        threads = None
        if str(get_device(device)) == "cpu":
            threads = max(1, available_cpus() // max(workers, 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    run_job, claim, job, args.device, args.smoke, overlays,
                    args.n_train, args.epochs, shard_root, threads,
                ): job
                for job in pending
            }
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    fut.result()
                    print(f"done {job}", flush=True)
                except Exception as e:
                    errors.append((job, str(e)))
                    print(f"FAIL {job}: {e}  (see {shard_root / job_tag(claim, job) / 'log.txt'})", flush=True)

    if errors:
        raise SystemExit(
            f"{len(errors)}/{len(pending)} jobs failed: {[job_tag(claim, j) for j, _ in errors][:6]}\n"
            f"finished shards are kept; rerun the same command with --resume {parent}"
        )
    if args.shard:
        done, missing = shard_status(claim, jobs, shard_root)
        print(f"shard {args.shard} finished; {len(done)}/{len(jobs)} of the grid is done. "
              f"Merge with: run_parallel.py {claim} --merge-only {parent}", flush=True)
        return
    write_merged(claim, parent, jobs, cfg, args.allow_partial, link=not args.no_latest, extra=extra)


if __name__ == "__main__":
    main()
