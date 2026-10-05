import json
import tempfile
import unittest
from pathlib import Path

from training.metrics import MetricsRecorder, read_metrics, restore_metric_offsets, write_json


class TestMetrics(unittest.TestCase):
    def test_append_does_not_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for decision in (1, 2):
                with MetricsRecorder(path) as recorder:
                    recorder.write("decisions", {"decision": decision, "epsilon": 0.5})
            self.assertEqual([row["decision"] for row in read_metrics(path, "decisions")], ["1", "2"])

    def test_recovery_archives_trailing_records_before_restoring_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path) as recorder:
                recorder.write("decisions", {"decision": 1})
                offsets = recorder.offsets()
                recorder.write("decisions", {"decision": 2})
            restore_metric_offsets(path, offsets)
            self.assertEqual([row["decision"] for row in read_metrics(path, "decisions")], ["1"])
            archive = next((path / "recoveries").glob("*/decisions.csv"))
            self.assertIn("2", archive.read_text())
            with MetricsRecorder(path) as recorder:
                recorder.write("decisions", {"decision": 2})
            self.assertEqual(len(read_metrics(path, "decisions")), 2)

    def test_invalid_recovery_offset_does_not_change_any_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path) as recorder:
                offsets = recorder.offsets()
            contents = {file.name: file.read_bytes() for file in path.glob("*.csv")}
            offsets["evaluation"] += 100
            with self.assertRaises(ValueError):
                restore_metric_offsets(path, offsets)
            self.assertEqual(contents, {file.name: file.read_bytes() for file in path.glob("*.csv")})

    def test_incompatible_csv_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "decisions.csv").write_text("wrong,header\n")
            with self.assertRaises(ValueError):
                MetricsRecorder(path)

    def test_metadata_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_json(path, {"status": "running", "episodes": 2})
            write_json(path, {"status": "completed", "episodes": 3})
            self.assertEqual(json.loads(path.read_text()), {"status": "completed", "episodes": 3})
