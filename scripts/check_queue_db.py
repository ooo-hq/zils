"""Validate the migration and queue transactions in a disposable local PostgreSQL cluster."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pg_bin = os.environ.get("FEZ_PG_BIN")

    def tool(name):
        path = str(Path(pg_bin) / name) if pg_bin else shutil.which(name)
        if not path or not Path(path).is_file():
            raise RuntimeError("Install PostgreSQL 16+ or set FEZ_PG_BIN to its bin directory")
        return path

    with tempfile.TemporaryDirectory(prefix="fez-queue-db-") as tmp:
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
