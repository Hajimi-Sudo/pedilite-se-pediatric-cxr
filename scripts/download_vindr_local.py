"""Download the VinDr-PCXR test set with authenticated parallel resume."""

from __future__ import annotations

import base64
import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://physionet.org/files/vindr-pcxr/1.0.0"
TEST_URL = f"{BASE_URL}/test"
CHUNK_SIZE = 1024 * 1024
RETRIES = 5


def auth_header() -> str:
    user = os.environ.get("VINDR_USER", "")
    password = os.environ.get("VINDR_PASSWORD", "")
    if not user or not password:
        raise RuntimeError("VINDR_USER and VINDR_PASSWORD must be set")
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


AUTH = ""


def request(url: str, headers: dict[str, str] | None = None):
    merged = {"User-Agent": "PediLiteSE-local-downloader/1.0"}
    if AUTH:
        merged["Authorization"] = AUTH
    if headers:
        merged.update(headers)
    return urlopen(Request(url, headers=merged), timeout=90)


def fetch_index() -> list[str]:
    global AUTH
    saved_auth = AUTH
    AUTH = ""
    try:
        with request(f"{TEST_URL}/") as response:
            html = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code != 401:
            raise
        AUTH = saved_auth
        with request(f"{TEST_URL}/") as response:
            html = response.read().decode("utf-8", errors="replace")
    else:
        AUTH = saved_auth
    names = sorted(set(re.findall(r'href="([0-9a-f]{32}\.dicom)"', html)))
    if len(names) != 1397:
        raise RuntimeError(f"expected 1397 DICOM links, found {len(names)}")
    return names


def download_file(url: str, destination: Path, expected_size: int | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        current = destination.stat().st_size if destination.exists() else 0
        if expected_size is not None and current == expected_size:
            return "exists"
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            with request(url, headers) as response:
                partial = current > 0 and response.status == 206
                mode = "ab" if partial else "wb"
                written = current if partial else 0
                with destination.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
            if expected_size is not None and written != expected_size:
                raise RuntimeError(f"size {written} != expected {expected_size}")
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def get_expected_sizes(names: list[str]) -> dict[str, int]:
    with request(f"{TEST_URL}/") as response:
        html = response.read().decode("utf-8", errors="replace")
    sizes: dict[str, int] = {}
    for name, size in re.findall(r'href="([0-9a-f]{32}\.dicom)"[^\n]*?([0-9]+)\s*$', html, re.MULTILINE):
        sizes[name] = int(size)
    return {name: sizes.get(name, 0) for name in names}


def main() -> int:
    global AUTH
    AUTH = auth_header()
    root = Path(os.environ.get("VINDR_DATA_DIR", "experiment/code/data/vindr_pcxr_test"))
    workers = int(os.environ.get("VINDR_WORKERS", "32"))
    root.mkdir(parents=True, exist_ok=True)

    names = fetch_index()
    sizes = get_expected_sizes(names)
    print(f"START files={len(names)} workers={workers} root={root}", flush=True)

    def job(name: str) -> tuple[str, str]:
        status = download_file(f"{TEST_URL}/{name}", root / name, sizes[name] or None)
        return name, status

    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, name) for name in names]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, status = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(names):
                    print(f"PROGRESS {completed}/{len(names)} last={name} status={status}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"ERROR {exc}", file=sys.stderr, flush=True)

    labels = root.parent / "image_labels_test.csv"
    try:
        download_file(f"{BASE_URL}/image_labels_test.csv", labels)
        print(f"LABELS {labels}", flush=True)
    except Exception as exc:
        print(f"LABEL_ERROR {exc}", file=sys.stderr, flush=True)
        failed += 1

    print(f"DONE completed={completed} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
