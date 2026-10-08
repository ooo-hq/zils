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

            run([tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"])
            migration = ROOT / "supabase/migrations/202610070001_early_access.sql"
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
                    str(migration),
                ],
                check=True,
                capture_output=True,
            )
            from tests.access_database import run as access_checks

            access_checks(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
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
                    str(ROOT / "supabase/migrations/202610080001_image_assets.sql"),
                ],
                check=True,
                capture_output=True,
            )
            image_checks(
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
                    str(ROOT / "supabase/migrations/202610080002_image_jobs.sql"),
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
                    str(ROOT / "supabase/migrations/202610080003_image_worker_profiles.sql"),
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
                    str(ROOT / "supabase/migrations/202610080004_image_processing_profiles.sql"),
                ],
                check=True,
                capture_output=True,
            )
            from tests.image_queue_database import cancellation_claim_races

            cancellation_claim_races(
                [tool("psql"), "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
            )
            from tests.image_processing_database import run as processing_checks

            processing_checks(
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
