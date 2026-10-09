"""Validate the migration and queue transactions in a disposable local PostgreSQL cluster."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from zils import settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    pg_bin = settings.get("ZILS_PG_BIN")

    def tool(name):
        path = str(Path(pg_bin) / name) if pg_bin else shutil.which(name)
        if not path or not Path(path).is_file():
            raise RuntimeError("Install PostgreSQL 16+ or set ZILS_PG_BIN to its bin directory")
        return path

    with tempfile.TemporaryDirectory(prefix="zils-queue-db-") as tmp:
        root = Path(tmp)
        data, socket = root / "data", root / "socket"
        socket.mkdir()
        subprocess.run(
            [tool("initdb"), "-D", str(data), "-A", "trust", "--no-locale"],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(
                [
                    tool("pg_ctl"),
                    "-D",
                    str(data),
                    "-l",
                    str(root / "postgres.log"),
                    "-o",
                    f"-k {socket} -c listen_addresses=''",
                    "-w",
                    "start",
                ],
                check=True,
                capture_output=True,
            )
            for source in (
                "tests/sql/queue-bootstrap.sql",
                "supabase/migrations/202609300001_training_jobs.sql",
                "tests/sql/queue-assertions.sql",
                "supabase/migrations/202610040001_decision_api.sql",
                "tests/sql/api-assertions.sql",
                "supabase/migrations/202610040002_decision_batches.sql",
                "tests/sql/batch-assertions.sql",
            ):
                result = subprocess.run(
                    [
                        tool("psql"),
                        "-X",
                        "-v",
                        "ON_ERROR_STOP=1",
                        "-h",
                        str(socket),
                        "-d",
                        "postgres",
                        "-f",
                        str(ROOT / source),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(f"{source}:\n{result.stdout}\n{result.stderr}")
            from tests.api_database import run

            # Verify the production billing/usage schema before adding image tables.
            for source in (
                "202610080001_prepaid_billing.sql",
                "202610080002_usage_reporting.sql",
            ):
                billing = ROOT / "supabase/migrations" / source
                result = subprocess.run(
                    [
                        tool("psql"),
                        "-X",
                        "-v",
                        "ON_ERROR_STOP=1",
                        "-h",
                        str(socket),
                        "-d",
                        "postgres",
                        "-f",
                        str(billing),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
            run([tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"])
            from tests.billing_database import run as run_billing
            from tests.billing_webhook_database import run as run_billing_webhooks
            from tests.usage_database import run as run_usage

            run_billing(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            run_billing_webhooks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            run_usage(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-c",
                    "update zils_billing_settings set mode='off'",
                ],
                check=True,
                capture_output=True,
            )
            from tests.image_database import run as image_checks

            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(ROOT / "supabase/migrations/202610080003_image_assets.sql"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(ROOT / "supabase/migrations/202610080004_image_jobs.sql"),
                ],
                check=True,
                capture_output=True,
            )
            from tests.image_job_database import run as image_job_checks

            image_job_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_queue_database import run as image_queue_checks

            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(ROOT / "supabase/migrations/202610080005_image_worker_profiles.sql"),
                ],
                check=True,
                capture_output=True,
            )
            image_queue_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(ROOT / "supabase/migrations/202610080006_image_processing_profiles.sql"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(ROOT / "supabase/migrations/202610080007_image_finalize_recovery.sql"),
                ],
                check=True,
                capture_output=True,
            )
            image_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_recovery_database import run as recovery_checks

            recovery_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_queue_database import cancellation_claim_races

            cancellation_claim_races(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_processing_database import run as processing_checks

            processing_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_billing_database import run as image_billing_checks
            from tests.storage_database import run as storage_checks

            storage_migration = ROOT / "supabase/migrations/202610090001_spaces_storage.sql"
            subprocess.run(
                [
                    tool("psql"),
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    str(socket),
                    "-d",
                    "postgres",
                    "-f",
                    str(storage_migration),
                ],
                check=True,
                capture_output=True,
            )
            storage_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )

            image_billing_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            # Image migrations replace queue functions: recheck production billing
            # after the full upgrade, including the cancellation/claim race.
            run_billing(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            run_billing_webhooks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            run_usage(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            print(
                "Training queue and decision API migrations, isolation, leases, credentials and admission passed."
            )
        finally:
            subprocess.run(
                [tool("pg_ctl"), "-D", str(data), "-m", "immediate", "-w", "stop"],
                capture_output=True,
            )


if __name__ == "__main__":
    main()
