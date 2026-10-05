"""Download a terminal Zils batch as JSONL using current account authorization."""

import argparse
import json
import os
from pathlib import Path

import requests

from zils.api_store import identifier
from zils.cloud import trusted_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8920")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    url = trusted_url(args.url) + "/v1/batches/" + identifier(args.batch) + "/results"
    headers = {"Authorization": "Bearer " + os.environ["ZILS_API_KEY"]}
    cursor, count = 0, 0
    # Exclusive creation prevents replacement of existing customer results.
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            while True:
                with requests.get(
                    url,
                    params={"after": cursor},
                    headers=headers,
                    timeout=(10, 60),
                    allow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    page = response.json()
                for record in page["data"]:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
                next_cursor = page["next_cursor"]
                if next_cursor is None:
                    break
                if type(next_cursor) is not int or next_cursor <= cursor:
                    raise ValueError("Invalid results cursor")
                cursor = next_cursor
    except BaseException:
        # An interrupted export must not look like a complete result file.
        args.out.unlink(missing_ok=True)
        raise
    print(f"Downloaded {count} committed records.")


if __name__ == "__main__":
    main()
