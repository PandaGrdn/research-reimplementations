#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz import plot_per_step_curves


def load_runs(results_dir):
    runs = []
    for d in sorted(Path(results_dir).iterdir()):
        metrics_path = d / "metrics.json"
        cfg_path = d / "config.yaml"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        name = d.name.split("_seed")[0]
        seed = None
        for part in d.name.split("_"):
            if part.startswith("seed"):
                try:
                    seed = int(part.replace("seed", ""))
                except ValueError:
                    pass
        runs.append({"dir": d, "name": name, "seed": seed, "metrics": metrics, "cfg": cfg_path})
    return runs


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def fmt(mu, sd):
    if mu is None:
        return "—"
    return f"{mu:.4f} ± {sd:.4f}"


def aggregate(runs):
    by_name = defaultdict(list)
    for r in runs:
        by_name[r["name"]].append(r["metrics"])
    latest = {}
    for r in runs:
        latest[r["name"]] = r["metrics"]
    return by_name, latest


def markdown_table(by_name, contexts=(2, 8), horizons=(1, 5, 10, 18)):
    lines = ["| strategy | context | h=1 AR MSE | h=5 | h=10 | h=18 | headline MSE 5–18 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(by_name):
        mets = by_name[name]
        for c in contexts:
            cs = str(c)
            cells = [name, str(c)]
            for h in horizons:
                vals = []
                for m in mets:
                    v = m.get("horizons", {}).get(cs, {}).get(str(h), {}).get("ar_mse")
                    if v is not None:
                        vals.append(v)
                mu, sd = mean_std(vals)
                cells.append(fmt(mu, sd) if mu is not None else "—")
            heads = [m.get("headline", {}).get(cs, {}).get("mse_5_18") for m in mets]
            mu, sd = mean_std(heads)
            cells.append(fmt(mu, sd))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "results" / "report"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    if not runs:
        print("no results/*/metrics.json found")
        return
    by_name, latest = aggregate(runs)
    table = markdown_table(by_name)
    (out / "table.md").write_text(table + "\n")
    print(table)
    plot_per_step_curves(latest, "mse", 2, out / "ar_mse_context2.png", ylabel="pixel MSE")
    plot_per_step_curves(latest, "mse", 8, out / "ar_mse_context8.png", ylabel="pixel MSE")
    plot_per_step_curves(latest, "latent_cos", 8, out / "latent_cos_context8.png", ylabel="cosine similarity")
    plot_per_step_curves(latest, "ssim", 8, out / "ar_ssim_context8.png", ylabel="SSIM")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
