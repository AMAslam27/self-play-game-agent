import json
import tempfile
import unittest
from pathlib import Path

from training.metrics import MetricsRecorder, read_metrics, restore_metric_offsets, write_json


class TestMetrics(unittest.TestCase):
    def test_opponent_metrics_are_optional_and_disabled_writes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path) as recorder:
                self.assertNotIn("opponents", recorder.offsets())
                with self.assertRaisesRegex(ValueError, "not enabled"):
                    recorder.write("opponents", {})
            self.assertFalse((path / "opponents.csv").exists())

    def test_opponent_and_episode_history_recover_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path, include_opponents=True) as recorder:
                for table in ("episodes", "opponents"):
                    recorder.write(table, {"session": 1, "episode": 1})
                offsets = recorder.offsets()
                for table in ("episodes", "opponents"):
                    recorder.write(table, {"session": 1, "episode": 2})
            restore_metric_offsets(path, offsets, include_opponents=True)
            for table in ("episodes", "opponents"):
                self.assertEqual([row["episode"] for row in read_metrics(path, table)], ["1"])
                self.assertTrue(list((path / "recoveries").glob(f"*/{table}.csv")))
            with MetricsRecorder(path, include_opponents=True) as recorder:
                recorder.write("opponents", {"session": 2, "episode": 2})
            self.assertEqual(len(read_metrics(path, "opponents")), 2)

    def test_missing_opponent_offset_fails_without_modifying_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path, include_opponents=True) as recorder:
                offsets = recorder.offsets()
                recorder.write("episodes", {"episode": 1})
            del offsets["opponents"]
            contents = {file.name: file.read_bytes() for file in path.glob("*.csv")}
            with self.assertRaises(ValueError):
                restore_metric_offsets(path, offsets, include_opponents=True)
            self.assertEqual(contents, {file.name: file.read_bytes() for file in path.glob("*.csv")})

    def test_bad_opponent_header_prevents_all_recovery_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with MetricsRecorder(path, include_opponents=True) as recorder:
                offsets = recorder.offsets()
                recorder.write("episodes", {"episode": 1})
            (path / "opponents.csv").write_text("wrong,header\n" + "x" * offsets["opponents"])
            contents = {file.name: file.read_bytes() for file in path.glob("*.csv")}
            with self.assertRaisesRegex(ValueError, "Unexpected CSV schema"):
                restore_metric_offsets(path, offsets, include_opponents=True)
            self.assertEqual(contents, {file.name: file.read_bytes() for file in path.glob("*.csv")})

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
