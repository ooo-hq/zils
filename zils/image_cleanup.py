"""Delete expired image objects using leased, retryable cleanup transactions."""

import argparse
import json
import time

from .image_store import ImageStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    store = ImageStore()
    while True:
        print(json.dumps(store.cleanup()), flush=True)
        if args.once:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
