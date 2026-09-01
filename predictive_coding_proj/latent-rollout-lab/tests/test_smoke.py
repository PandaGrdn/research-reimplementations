import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_config
from src.data import cache_latents, load_splits, sequence_loader, split_hash
from src.evaluate import evaluate
from src.models.autoencoder import build_ae
from src.models.registry import STRATEGIES, build_model, build_strategy
from src.pretrain_ae import train_ae
from src.train import train_temporal
from src.utils import get_device, seed_everything


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.root = ROOT
        cls.device = get_device("cpu")
        cfg = resolve_config(ROOT / "configs" / "strategy" / "smoke.yaml", root=ROOT)
        cfg["eval"]["lpips"] = False
        cfg["ae"]["epochs"] = 1
        cfg["temporal"]["epochs"] = 1
        cfg["data"]["n_train"] = 32
        cfg["data"]["n_val"] = 8
        cfg["data"]["n_test"] = 8
        data_root = (ROOT / cfg["data"]["root"]).resolve()
        train, val, test, info = load_splits(
            data_root, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"], 0
        )
        cls.cfg = cfg
        cls.train, cls.val, cls.test, cls.info = train, val, test, info
        cls.ae = build_ae(cfg).to(cls.device)
        train_ae(cls.ae, train, val, cfg, cls.device, log=lambda *_: None)
        for p in cls.ae.parameters():
            p.requires_grad = False
        cls.ae.eval()
        cls.train_lat = cache_latents(cls.ae, train, cls.device)
        cls.val_lat = cache_latents(cls.ae, val, cls.device)
        cls.test_lat = cache_latents(cls.ae, test, cls.device)
        import torch

        cls.train_fr = torch.stack(train)
        cls.val_fr = torch.stack(val)
        cls.test_fr = torch.stack(test)

    def test_split_hash_stable(self):
        data_root = (ROOT / self.cfg["data"]["root"]).resolve()
        train2, val2, test2, _ = load_splits(
            data_root, self.cfg["data"]["n_train"], self.cfg["data"]["n_val"], self.cfg["data"]["n_test"], 0
        )
        self.assertEqual(split_hash(self.train, self.val, self.test), split_hash(train2, val2, test2))
        self.assertEqual(tuple(self.train[0].shape), (20, 1, 64, 64))

    def test_strategies(self):
        import torch

        for name in STRATEGIES:
            cfg = dict(self.cfg)
            cfg["strategy"] = {"name": name, "warmup": 0, "epochs_per_stage": 1, "max_unroll": 2}
            if name == "scheduled_sampling":
                cfg["strategy"].update({"p_start": 0.5, "p_end": 0.5, "include_tf_loss": True})
            model = build_model(cfg).to(self.device)
            strategy = build_strategy(cfg)
            out = strategy.compute_loss(
                model, self.ae, self.train_lat[:4].to(self.device), self.train_fr[:4].to(self.device), 0
            )
            self.assertTrue(torch.isfinite(out["loss"]))
            train_temporal(
                model,
                self.ae,
                self.train_fr,
                self.train_lat,
                self.val_fr,
                self.val_lat,
                strategy,
                cfg,
                self.device,
                log=lambda *_: None,
                run_dir=None,
            )
            loader = sequence_loader(self.test_fr, self.test_lat, 4, shuffle=False)
            metrics = evaluate(
                model, self.ae, loader, context_lengths=(2, 8), horizons=(1, 5), lpips=False, device=self.device
            )
            self.assertIn("ae_recon_mse", metrics)
            self.assertTrue(metrics["ar"]["8"]["mse"])
            json.dumps(metrics)


if __name__ == "__main__":
    unittest.main()
