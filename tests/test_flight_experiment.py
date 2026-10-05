"""Protect flight experiment labels, temporal separation, and probability scoring."""

import copy
import csv
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from scripts import flight_data as data, flight_experiment as experiment


def source(**changes):
    return {
        "FlightDate": "2025-01-02",
        "Reporting_Airline": "AA",
        "Origin": "JFK",
        "Dest": "LAX",
        "Flight_Number_Reporting_Airline": "123",
        "CRSDepTime": "0900",
        "CRSArrTime": "1200",
        "CRSElapsedTime": "360",
        "Distance": "2475",
        "Cancelled": "0",
        "Diverted": "0",
        "ArrDelay": "15",
        "ArrDel15": "1",
        "DepDelay": "900",
        "ActualElapsedTime": "999",
        **changes,
    }


class FlightExperimentTest(unittest.TestCase):
    def test_live_run_requires_capacity_floor_and_unmodified_base(self):
        environment = {"FEZ_COMPUTE_LOCK": "/tmp/test.lock", "FEZ_GPU_MIN_FREE_MIB": "12288"}
        metadata = {"kind": "base", "temperature": 1.22}
        experiment.validate_run(environment, metadata)
        for value in ("1", "8192", "0"):
            with self.assertRaises(ValueError):
                experiment.validate_run({**environment, "FEZ_GPU_MIN_FREE_MIB": value}, metadata)
        for value in ({"kind": "adapter", "temperature": 1.22}, {"kind": "base", "temperature": 2}):
            with self.assertRaises(ValueError):
                experiment.validate_run(environment, value)

    @unittest.skipUnless(os.name == "posix", "GPU runner requires Linux/WSL")
    def test_terminating_parent_reaps_active_child_before_releasing_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            child = (
                "import os,time,pathlib;pathlib.Path("
                + repr(str(pid_file))
                + ").write_text(str(os.getpid()));time.sleep(30)"
            )
            outer = (
                "import os,sys;from pathlib import Path;from scripts.flight_experiment import execute;"
                "execute([sys.executable,'-c',"
                + repr(child)
                + "],Path("
                + repr(tmp + "/child.log")
                + "),dict(os.environ),Path.cwd(),30)"
            )
            parent = subprocess.Popen([sys.executable, "-c", outer])
            try:
                deadline = time.monotonic() + 5
                while (
                    not pid_file.exists() and time.monotonic() < deadline and parent.poll() is None
                ):
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "Child did not start")
                pid = int(pid_file.read_text())
                parent.terminate()
                parent.wait(timeout=8)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()

    def test_post_departure_fields_can_change_only_the_label(self):
        late = data.case(source())
        early = data.case(source(ArrDelay="-3", ArrDel15="0", DepDelay="-5"))
        self.assertEqual(late["state"], early["state"])
        self.assertEqual(late["id"], early["id"])
        self.assertEqual((late["label"], early["label"]), ("late", "not_late"))
        self.assertEqual(set(late["state"]), set(data.INPUT_FIELDS))
        self.assertNotIn("900", str(late["state"]))
        self.assertIsNone(data.case(source(Cancelled="1")))
        self.assertIsNone(data.case(source(Diverted="1")))
        self.assertIsNone(data.case(source(ArrDelay="", ArrDel15="")))
        with self.assertRaises(ValueError):
            data.case(source(ArrDelay="14", ArrDel15="1"))

    def test_archive_sampling_is_reproducible_and_rejects_wrong_month_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "flights.zip"

            def write(rows):
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("flights.csv", stream.getvalue())

            rows = [source(Flight_Number_Reporting_Airline=str(i)) for i in range(12)]
            write(rows)
            first, _ = data.sample_archive(archive, 1, 5)
            write(list(reversed(rows)))
            second, _ = data.sample_archive(archive, 1, 5)
            self.assertEqual(first, second)
            with self.assertRaises(ValueError):
                data.sample_archive(archive, 2, 5)
            write([*rows, rows[0]])
            with self.assertRaises(ValueError):
                data.sample_archive(archive, 1, 5)

    def test_historical_rates_are_fitted_only_from_supplied_training_rows(self):
        rows = [data.case(source(Flight_Number_Reporting_Airline=str(i))) for i in range(4)]
        rows[0]["label"] = "not_late"
        baseline = data.fit_history(rows)
        unseen = copy.deepcopy(rows[0])
        unseen["state"]["airline"] = "UNKNOWN"
        self.assertAlmostEqual(data.history_probability(baseline, unseen), 4 / 6)
        before = data.history_probability(baseline, rows[0])
        rows[0]["label"] = "late"
        self.assertEqual(data.history_probability(baseline, rows[0]), before)

    def test_metrics_use_binary_brier_and_handle_tied_ranks(self):
        values = experiment.metrics([0, 1], [0.25, 0.75])
        self.assertAlmostEqual(values["brier"], 0.0625)
        self.assertEqual(values["accuracy"], 1)
        self.assertEqual(values["roc_auc"], 1)
        self.assertEqual(experiment.metrics([0, 1], [0.5, 0.5])["roc_auc"], 0.5)
        for invalid in ([float("nan"), 0.5], [-0.1, 0.5], [0.5]):
            with self.assertRaises(ValueError):
                experiment.metrics([0, 1], invalid)

    def test_temperature_fitting_uses_calibration_only(self):
        temperature = experiment.fit_temperature([0, 1, 0, 1], [3, -3, 3, -3])
        self.assertEqual(temperature, 4)
        self.assertLess(experiment.probability(3, temperature), experiment.probability(3, 1))
        self.assertAlmostEqual(experiment.probability(-1000, 1), 0)
        self.assertAlmostEqual(experiment.probability(1000, 1), 1)


if __name__ == "__main__":
    unittest.main()
