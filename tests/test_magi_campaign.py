import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import magi_campaign


class MagiCampaignTests(unittest.TestCase):
    def test_dry_run_does_not_launch_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "campaign"
            args = magi_campaign.parser().parse_args(["--out-dir", str(folder), "--baseline", directory,
                "--gpu-ids", "5,6"])
            stream = io.StringIO()
            with patch.object(magi_campaign.queue_video, "execute") as execute, contextlib.redirect_stdout(stream):
                self.assertEqual(magi_campaign.run(args), 0)
            execute.assert_not_called()
            self.assertFalse(folder.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["planned_tiny_videos"], 0)
            self.assertEqual(plan["allocation"], "uniform")
            self.assertFalse(plan["arc_compensation"])

    def test_uniform_no_arc_campaign_rejects_formal_tiny_before_preflight(self):
        for execute_flag in ([], ["--execute"]):
            with self.subTest(execute=bool(execute_flag)), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / "campaign"
                args = magi_campaign.parser().parse_args(["--out-dir", str(folder),
                    "--baseline", directory, "--tiny-after-validation", *execute_flag])
                with patch.object(magi_campaign.queue_video, "build_plan") as build_plan:
                    with self.assertRaisesRegex(ValueError, "ARC compensation and learned greedy"):
                        magi_campaign.run(args)
                build_plan.assert_not_called()
                self.assertFalse(folder.exists())

    def test_failed_baseline_never_starts_sparse_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            baseline.mkdir()
            (baseline / "status.json").write_text(json.dumps(dict(state="failed", error="OOM")))
            output = Path(directory) / "campaign"
            args = magi_campaign.parser().parse_args(["--out-dir", str(output), "--baseline", str(baseline), "--execute"])
            with patch.object(magi_campaign.queue_video, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "Baseline failed"):
                    magi_campaign.run(args)
            execute.assert_not_called()
            self.assertEqual(json.loads((output / "campaign_status.json").read_text())["state"], "failed")

    def test_full_grid_has_only_requested_rates(self):
        self.assertEqual(magi_campaign.GRID, (.5, .6, .7, .8, .9))
        args = magi_campaign.parser().parse_args(["--out-dir", "/tmp/magi-test", "--baseline", "/tmp/baseline"])
        options = magi_campaign.queue_args(args, "wina", .7)
        self.assertEqual((options.model, options.quant, options.steps, options.refiner_steps),
                         ("magi-2-preview", "hifp4", 50, 5))
        self.assertEqual(options.gpu_ids, "5,6")


if __name__ == "__main__":
    unittest.main()
