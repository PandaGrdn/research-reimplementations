import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_parallel as rp
from src.utils import link_latest, new_run_dir


class ParallelTests(unittest.TestCase):
    def test_c2_job_count(self):
        jobs = rp.c2_jobs({}, [0, 1, 2], None)
        self.assertEqual(len(jobs), 9 * 3)
        self.assertEqual(jobs[0], {"cell": "baseline", "seed": 0})

    def test_c2_job_subset(self):
        jobs = rp.c2_jobs({}, [0], ["baseline", "split_fix"])
        self.assertEqual([j["cell"] for j in jobs], ["baseline", "split_fix"])

    def test_c7_job_count(self):
        cfg = {"c7": {"lambda_slow": [0.0, 1.0], "unroll_k": [0, 5]}}
        jobs = rp.c7_jobs(cfg, [0, 1])
        self.assertEqual(len(jobs), 8)
        jobs = rp.c7_jobs(cfg, [0], lambdas=[0.0], ks=[5])
        self.assertEqual(jobs, [{"unroll_k": 5, "lambda_slow": 0.0, "seed": 0}])

    def test_merge_c2_averages_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i, (name, val) in enumerate([("baseline", 0.4), ("baseline", 0.6), ("split_fix", 0.5)]):
                d = td / f"s{i}"
                d.mkdir()
                (d / "metrics.json").write_text(json.dumps({
                    "headline_split": 10,
                    "cells": [{
                        "name": name,
                        "long_mse_mean": val,
                        "long_mse_std": 0.0,
                        "copy_last_long_mse_mean": val - 0.1,
                        "mean_frame_long_mse_mean": val - 0.05,
                        "per_seed": [val],
                    }],
                }))
            merged = rp.merge_c2(sorted(td.iterdir()))
            by = {c["name"]: c for c in merged["cells"]}
            self.assertAlmostEqual(by["baseline"]["long_mse_mean"], 0.5)
            self.assertEqual(by["baseline"]["per_seed"], [0.4, 0.6])
            self.assertAlmostEqual(by["split_fix"]["long_mse_mean"], 0.5)

    def test_merge_c7_concatenates_per_seed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i, val in enumerate([0.2, 0.4]):
                d = td / f"s{i}"
                d.mkdir()
                (d / "metrics.json").write_text(json.dumps({
                    "sweep": [{
                        "unroll_k": 0,
                        "lambda_slow": 1.0,
                        "cons_cos_no_pull": val,
                        "per_seed": {
                            "seeds": [i],
                            "cons_cos_no_pull": [val],
                            "frac_rel_no_pull": [0.8],
                            "cons_cos_with_pull": [val + 0.1],
                            "frac_rel_with_pull": [0.7],
                            "dict_drift": [0.01],
                            "val_mse": [0.3],
                            "copy_last_mse": [0.35],
                            "grad_path": [False],
                        },
                    }],
                }))
            merged = rp.merge_c7(sorted(td.iterdir()))
            rec = merged["sweep"][0]
            self.assertAlmostEqual(rec["cons_cos_no_pull"], 0.3)
            self.assertEqual(rec["per_seed"]["cons_cos_no_pull"], [0.2, 0.4])
            self.assertFalse(rec["grad_path"])

    def test_new_run_dir_skip_latest(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "shard"
            os.environ["PC_LAB_RUN_DIR"] = str(shard)
            os.environ["PC_LAB_SKIP_LATEST"] = "1"
            try:
                out = new_run_dir(td, "c2_mitigation_grid")
                self.assertEqual(out, shard)
                self.assertFalse((td / "results" / "c2_mitigation_grid" / "latest").exists())
            finally:
                os.environ.pop("PC_LAB_RUN_DIR", None)
                os.environ.pop("PC_LAB_SKIP_LATEST", None)

    def test_link_latest_points_at_parent(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            run = td / "results" / "c2_mitigation_grid" / "20260101_000000"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text("{}")
            link_latest(td, "c2_mitigation_grid", run)
            latest = td / "results" / "c2_mitigation_grid" / "latest"
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), run.resolve())

    def test_write_job_overlay_tags(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path, tag = rp.write_job_overlay(td, "c2", {"cell": "baseline", "seed": 1})
            self.assertEqual(tag, "baseline_s1")
            self.assertTrue(path.exists())
            path, tag = rp.write_job_overlay(td, "c7", {"unroll_k": 5, "lambda_slow": 0.5, "seed": 2})
            self.assertEqual(tag, "k5_l0.5_s2")


def _fake_c2_shard(shard_root, cell, seed, val):
    d = Path(shard_root) / f"{cell}_s{seed}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "headline_split": 10,
        "cells": [{
            "name": cell,
            "long_mse_mean": val,
            "long_mse_std": 0.0,
            "copy_last_long_mse_mean": 0.07,
            "mean_frame_long_mse_mean": 0.05,
            "per_seed": [val],
            "seeds": [seed],
        }],
    }))
    return d


class MergeGuardTests(unittest.TestCase):
    def test_warmup_cfg_has_the_jobs_dictionary_key(self):
        """C7 flips spatial.freeze and --n-train changes data.n_train; both are in
        the dictionary cache key, so the warmup must apply them too."""
        from src.config import resolve_config
        from src.spatial_pc import dictionary_key

        cfg = resolve_config(smoke=True, root=ROOT)
        as_c7_job = json.loads(json.dumps(cfg))
        as_c7_job["spatial"]["freeze"] = False  # what c7_slowness_sweep.main does
        as_c7_job["data"]["n_train"] = 7        # what setup() does for --n-train 7
        warm = rp.job_base_cfg(cfg, "c7", n_train=7)
        self.assertEqual(dictionary_key(warm, seed=1), dictionary_key(as_c7_job, seed=1))
        self.assertNotEqual(dictionary_key(cfg, seed=1), dictionary_key(as_c7_job, seed=1))
        self.assertEqual(dictionary_key(rp.job_base_cfg(cfg, "c2"), seed=1), dictionary_key(cfg, seed=1))

    def test_merge_refuses_incomplete_grid(self):
        jobs = rp.c2_jobs({}, [0, 1], ["baseline", "split_fix"])
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            for job, val in zip(jobs[:-1], [0.4, 0.6, 0.5]):
                _fake_c2_shard(parent / "shards", job["cell"], job["seed"], val)
            with self.assertRaises(SystemExit) as cm:
                rp.write_merged("c2", parent, jobs, {}, link=False)
            self.assertIn("split_fix_s1", str(cm.exception))
            self.assertFalse((parent / "metrics.json").exists())
            m = rp.write_merged("c2", parent, jobs, {}, allow_partial=True, link=False)
            self.assertEqual(m["missing_shards"], ["split_fix_s1"])
            self.assertIn("PARTIAL", m["summary"])
            self.assertFalse(m["seeds_consistent_across_cells"])

    def test_merge_keeps_job_order_and_seed_labels(self):
        jobs = rp.c2_jobs({}, [0, 1, 2], ["baseline"])
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            for seed, val in [(2, 0.3), (0, 0.1), (1, 0.2)]:  # written out of order
                _fake_c2_shard(parent / "shards", "baseline", seed, val)
            m = rp.write_merged("c2", parent, jobs, {}, link=False)
            cell = m["cells"][0]
            self.assertEqual(cell["seeds"], [0, 1, 2])
            self.assertEqual(cell["per_seed"], [0.1, 0.2, 0.3])
            self.assertIn("copy-last floor", m["summary"])

    def test_merge_rejects_double_counted_seed(self):
        with tempfile.TemporaryDirectory() as td:
            a = _fake_c2_shard(Path(td) / "a", "baseline", 0, 0.1)
            b = _fake_c2_shard(Path(td) / "b", "baseline", 0, 0.1)
            with self.assertRaises(SystemExit):
                rp.merge_c2([a, b])

    def test_merge_refuses_mixed_dictionaries(self):
        jobs = rp.c2_jobs({}, [0], ["baseline", "split_fix"])
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            for job, fp in zip(jobs, ["aaaa", "bbbb"]):
                d = _fake_c2_shard(parent / "shards", job["cell"], job["seed"], 0.4)
                (d / "env.json").write_text(json.dumps({"dictionary_fingerprints": {"0": fp}}))
            with self.assertRaises(SystemExit) as cm:
                rp.write_merged("c2", parent, jobs, {}, link=False)
            self.assertIn("different dictionaries", str(cm.exception))

    def test_model_refuses_other_dictionary(self):
        import torch
        from src import experiment as ex
        from src.config import resolve_config
        from src.spatial_pc import make_from_cfg

        cfg = resolve_config(smoke=True, root=ROOT)
        _, r_init, _ = make_from_cfg(cfg, "cpu")
        model = ex.build_temporal(cfg, r_init, "cpu")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "temporal.pt"
            ex.DICT_FINGERPRINTS[7] = "trained-on-this"
            ex.save_temporal_checkpoint(path, model, cfg, seed=7)
            ex.load_temporal_checkpoint(path, r_init, cfg, "cpu")  # same dictionary: fine
            ex.DICT_FINGERPRINTS[7] = "a-different-one"
            with self.assertRaises(SystemExit):
                ex.load_temporal_checkpoint(path, r_init, cfg, "cpu")
            ex.DICT_FINGERPRINTS.pop(7)
        self.assertTrue(torch.is_tensor(r_init[0]))

    def test_shard_needs_resume_and_partitions_jobs(self):
        with self.assertRaises(SystemExit):
            rp.main(["c2", "--smoke", "--shard", "0/3"])
        jobs = rp.c2_jobs({}, [0, 1, 2], None)
        parts = [jobs[i::5] for i in range(5)]
        self.assertEqual(sorted(map(str, sum(parts, []))), sorted(map(str, jobs)))


class EndToEndTests(unittest.TestCase):
    """Real smoke-scale subprocess runs (about 30 s). Each uses its own dictionary
    cache dir, so both sides start cold and nothing in results/ or artifacts/ is touched."""

    def _overlay(self, td, name):
        from src.utils import write_yaml

        path = Path(td) / f"{name}.yaml"
        write_yaml(path, {"spatial": {"checkpoint": str(Path(td) / name / "dictionary.pt")}})
        return str(path)

    def _sequential(self, td, claim, seeds):
        run_dir = Path(td) / f"seq_{claim}"
        env = os.environ.copy()
        env.update(PC_LAB_RUN_DIR=str(run_dir), PC_LAB_SKIP_LATEST="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        script = ROOT / "experiments" / f"{rp.EXP_NAME[claim]}.py"
        subprocess.check_call(
            [sys.executable, str(script), "--smoke", "--device", "cpu", "--seeds", *map(str, seeds),
             "--overlay", self._overlay(td, f"cache_seq_{claim}")],
            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        return json.loads((run_dir / "metrics.json").read_text())

    def _parallel(self, td, claim, seeds, extra=()):
        parent = Path(td) / f"par_{claim}"
        old = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            rp.main([claim, "--smoke", "--device", "cpu", "--workers", "4", "--no-latest",
                     "--seeds", *map(str, seeds), "--resume", str(parent),
                     "--overlay", self._overlay(td, f"cache_par_{claim}"), *extra])
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return parent, json.loads((parent / "metrics.json").read_text())

    def test_c2_parallel_equals_sequential_and_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            seq = self._sequential(td, "c2", [0, 1])
            parent, par = self._parallel(td, "c2", [0, 1])
            self.assertEqual([c["name"] for c in seq["cells"]], [c["name"] for c in par["cells"]])
            for a, b in zip(seq["cells"], par["cells"]):
                self.assertEqual(a["per_seed"], b["per_seed"], a["name"])
                self.assertEqual(a["seeds"], b["seeds"])
                self.assertEqual(a["long_mse_std"], b["long_mse_std"])
            self.assertEqual(seq["summary"], par["summary"])
            self.assertEqual(par["missing_shards"], [])

            # Lose one shard, resume: only that job reruns and the merge is unchanged.
            shards = parent / "shards"
            victim = shards / "baseline_s1"
            keep = shards / "all_combined_s0" / "metrics.json"
            (victim / "metrics.json").unlink()
            stamp = keep.stat().st_mtime_ns
            with self.assertRaises(SystemExit):
                rp.main(["c2", "--smoke", "--seeds", "0", "1", "--merge-only", str(parent), "--no-latest"])
            _, again = self._parallel(td, "c2", [0, 1])
            self.assertEqual(keep.stat().st_mtime_ns, stamp)
            self.assertTrue((victim / "metrics.json").exists())
            self.assertEqual([c["per_seed"] for c in again["cells"]], [c["per_seed"] for c in par["cells"]])

    def test_c7_parallel_equals_sequential_and_warmup_hits(self):
        with tempfile.TemporaryDirectory() as td:
            seq = self._sequential(td, "c7", [0, 1])
            parent, par = self._parallel(td, "c7", [0, 1])
            key = lambda r: (r["unroll_k"], r["lambda_slow"])
            self.assertEqual([key(r) for r in seq["sweep"]], [key(r) for r in par["sweep"]])
            for a, b in zip(seq["sweep"], par["sweep"]):
                self.assertEqual(a["per_seed"], b["per_seed"], key(a))
            for k in ("plateau_span_k0", "plateau_span_k5", "note", "summary"):
                self.assertEqual(seq[k], par[k])
            # Every job must load the dictionary the warmup cached, never retrain it.
            logs = sorted((parent / "shards").glob("*/log.txt"))
            self.assertEqual(len(logs), 8)
            for log in logs:
                text = log.read_text()
                self.assertIn("loaded dictionary", text, log)
                self.assertNotIn("saved dictionary", text, log)
            cached = list((Path(td) / "cache_par_c7").glob("dictionary_*.pt"))
            self.assertEqual(len(cached), 2)  # one per seed

    def _assert_same_metrics(self, seq, par):
        """Every key the sequential run wrote must be identical in the merge
        (compared as JSON so NaN == NaN)."""
        for k in seq:
            self.assertEqual(
                json.dumps(seq[k], sort_keys=True), json.dumps(par[k], sort_keys=True), f"metrics[{k!r}] differs"
            )

    def test_c1_per_seed_parallel_equals_sequential(self):
        with tempfile.TemporaryDirectory() as td:
            seq = self._sequential(td, "c1", [0, 1])
            parent, par = self._parallel(td, "c1", [0, 1])
            self.assertEqual([r["seed"] for r in par["seeds"]], [0, 1])
            self._assert_same_metrics(seq, par)
            self.assertIn("n_seeds=2", par["summary"])
            # per-seed checkpoints reachable under the same names as a sequential run
            self.assertTrue((parent / "seed0" / "temporal.pt").exists())
            self.assertTrue((parent / "seed1" / "temporal.pt").exists())

    def test_c8_per_seed_parallel_equals_sequential(self):
        with tempfile.TemporaryDirectory() as td:
            seq = self._sequential(td, "c8", [0, 1])
            _, par = self._parallel(td, "c8", [0, 1])
            self._assert_same_metrics(seq, par)

    def test_c3_parallel_and_from_c1_equal_trained_c3(self):
        with tempfile.TemporaryDirectory() as td:
            seq = self._sequential(td, "c3", [0, 1])
            _, par = self._parallel(td, "c3", [0, 1])
            self._assert_same_metrics(seq, par)

            # C3 read out of a C1 run must equal C3 trained on its own.
            cache = self._overlay(td, "cache_c1_for_c3")
            env = os.environ.copy()
            env.update(PC_LAB_SKIP_LATEST="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            c1_dir, derived_dir = Path(td) / "c1_run", Path(td) / "c3_from_c1"
            common = ["--smoke", "--device", "cpu", "--seeds", "0", "1", "--overlay", cache]
            subprocess.check_call(
                [sys.executable, str(ROOT / "experiments" / "c1_rollout_collapse.py"), *common],
                cwd=str(ROOT), env={**env, "PC_LAB_RUN_DIR": str(c1_dir)},
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            c3 = [sys.executable, str(ROOT / "experiments" / "c3_copy_detection.py"), "--from-c1", str(c1_dir)]
            subprocess.check_call(
                [*c3, *common], cwd=str(ROOT), env={**env, "PC_LAB_RUN_DIR": str(derived_dir)},
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            derived = json.loads((derived_dir / "metrics.json").read_text())
            self._assert_same_metrics(seq, derived)
            self.assertIn("derived_from_c1", derived)

            # A C1 run trained under a different config must be refused, as must a missing seed.
            for bad in (["--smoke", "--device", "cpu", "--seeds", "0", "--overlay", cache, "--epochs", "3"],
                        ["--smoke", "--device", "cpu", "--seeds", "0", "5", "--overlay", cache]):
                r = subprocess.run([*c3, *bad], cwd=str(ROOT), capture_output=True, text=True,
                                   env={**env, "PC_LAB_RUN_DIR": str(Path(td) / "c3_bad")})
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("--from-c1", r.stderr)
            self.assertFalse((Path(td) / "c3_bad" / "metrics.json").exists())

    def _run_script(self, td, claim, run_dir, extra_env=None, epochs="3"):
        env = os.environ.copy()
        env.update(PC_LAB_RUN_DIR=str(run_dir), PC_LAB_SKIP_LATEST="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("PC_LAB_TEST_STOP_AFTER_EPOCH", None)
        env.update(extra_env or {})
        script = ROOT / "experiments" / f"{rp.EXP_NAME[claim]}.py"
        return subprocess.run(
            [sys.executable, str(script), "--smoke", "--device", "cpu", "--seeds", "0", "--epochs", epochs,
             "--overlay", self._overlay(td, f"cache_resume_{claim}")],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        )

    def test_killed_training_resumes_bit_identically(self):
        """A job killed after epoch 1 of 3 and rerun in the same directory must
        produce exactly the metrics of an uninterrupted run (C1: plain training,
        C7: the dictionary trains too, C8: several training calls and an encoder)."""
        for claim in ("c1", "c7", "c8"):
            with self.subTest(claim=claim), tempfile.TemporaryDirectory() as td:
                full = Path(td) / "full"
                r = self._run_script(td, claim, full)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                cut = Path(td) / "cut"
                r = self._run_script(td, claim, cut, {"PC_LAB_TEST_STOP_AFTER_EPOCH": "1"})
                self.assertEqual(r.returncode, 75, r.stderr[-2000:])
                self.assertTrue((cut / "train_ckpt" / "call1.pt").exists())
                self.assertFalse((cut / "metrics.json").exists())
                r = self._run_script(td, claim, cut)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertIn("resumed training from", r.stdout)
                a = json.loads((full / "metrics.json").read_text())
                b = json.loads((cut / "metrics.json").read_text())
                self._assert_same_metrics(a, b)

    def test_shards_fill_one_run_dir_then_merge(self):
        with tempfile.TemporaryDirectory() as td:
            parent = None
            for i in range(2):
                old = os.environ.get("OMP_NUM_THREADS")
                rp.main(["c2", "--smoke", "--device", "cpu", "--workers", "1", "--seeds", "0",
                         "--cells", "baseline", "all_combined", "--resume", str(Path(td) / "run"),
                         "--shard", f"{i}/2", "--overlay", self._overlay(td, "cache_shard")])
                parent = Path(td) / "run"
                self.assertFalse((parent / "metrics.json").exists())
            rp.main(["c2", "--smoke", "--seeds", "0", "--cells", "baseline", "all_combined",
                     "--merge-only", str(parent), "--no-latest"])
            m = json.loads((parent / "metrics.json").read_text())
            self.assertEqual([c["name"] for c in m["cells"]], ["baseline", "all_combined"])


if __name__ == "__main__":
    unittest.main()
