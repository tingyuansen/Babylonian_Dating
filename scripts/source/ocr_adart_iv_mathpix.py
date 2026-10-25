#!/usr/bin/env python3
"""Submit, monitor, and download the canonical ADART IV Mathpix OCR.

The Mathpix Files API needs only the server-side API key stored as ``MATHPIX``
in ``~/.env``. The source SHA-256 is used as the idempotency key, so retrying a
live or completed submission cannot create a second billable job. The script
never prints or stores the credential. It keeps the line-level JSON, the only
output the pipeline reads, and an audit manifest. Mathpix's other formats hold
crops of the printed pages, which are under copyright, so none is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parents[2]
# The scan is under copyright and not distributed. A reader with the book puts
# it here, and git ignores it.
DEFAULT_SOURCE = HERE / "data" / "source" / "adart_iv" / "scan.pdf"
DEFAULT_OUT = HERE / "data" / "source" / "adart_iv"
API = "https://api.mathpix.com/files/v1"
# Each page of the scan carries at its foot the e-mail address of the account
# that downloaded the book. It is personal data and not part of the book, so
# the line-level OCR keeps this placeholder in its place.
# Mathpix reads the address both in plain and in full-width characters.
EMAIL = re.compile(r"[\w.+-]+[@＠][\w-]+(?:[.．][\w-]+)+")
REMOVED = "[e-mail address removed]"
# Mathpix links each figure region to a crop of the scanned page on its server.
# The crops are page images, so the links are removed.
IMAGE = re.compile(r"!\[[^\]]*\]\(https://cdn\.mathpix\.com/[^)]*\)"
                   r"|\\includegraphics(?:\[[^\]]*\])?\{https://cdn\.mathpix\.com/[^}]*\}")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 without loading a large PDF into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mathpix_key(env_path: Path) -> str:
    """Load only MATHPIX from a dotenv file without evaluating shell code."""
    if os.environ.get("MATHPIX"):
        return os.environ["MATHPIX"]
    pattern = re.compile(r"^\s*(?:export\s+)?MATHPIX\s*=\s*(.*?)\s*$")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip("\"'")
    raise RuntimeError(f"MATHPIX is absent from {env_path}")


def request_json(response: requests.Response) -> dict:
    """Raise with a redacted API error and return a JSON object."""
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Mathpix returned HTTP {response.status_code} and non-JSON content"
        ) from exc
    if not response.ok:
        raise RuntimeError(
            f"Mathpix returned HTTP {response.status_code}: "
            f"{json.dumps(body, sort_keys=True)[:1000]}"
        )
    return body


def submit(source: Path, out: Path, headers: dict[str, str]) -> dict:
    """Submit one idempotent direct-upload job and write the first manifest."""
    source_hash = sha256(source)
    options = {"metadata": {"improve_mathpix": False}}
    with source.open("rb") as stream:
        response = requests.post(
            API,
            headers={**headers, "Idempotency-Key": f"adart-iv-{source_hash}"},
            files={"file": (source.name, stream, "application/pdf")},
            data={"options_json": json.dumps(options, separators=(",", ":"))},
            timeout=600,
        )
    body = request_json(response)
    file_id = body.get("file_id")
    if not file_id:
        raise RuntimeError(f"Mathpix response has no file_id: {body}")
    manifest = {
        "schema": "adart-iv-mathpix-ocr-v1",
        "source": str(source.relative_to(HERE)),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "source_pages": 669,
        "mathpix_api": "Files API /files/v1 direct upload",
        "mathpix_file_id": file_id,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "submission_options": options,
        "status": "submitted",
        "outputs": {},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "mathpix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"submitted {source.name}: {file_id}")
    return manifest


def load_manifest(out: Path) -> dict:
    """Read the local job manifest required for polling or download."""
    path = out / "mathpix_manifest.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run with --submit first")
    return json.loads(path.read_text(encoding="utf-8"))


def poll(manifest: dict, out: Path, headers: dict[str, str], interval: int) -> dict:
    """Poll until OCR and every requested conversion are ready.

    Mathpix can mark the primary MMD OCR ``completed`` while a requested PDF
    conversion is still ``processing``. Treating that intermediate state as
    final produces a misleading 404 ``format_not_ready`` at download time.
    """
    file_id = manifest["mathpix_file_id"]
    last = None
    while True:
        body = request_json(requests.get(
            f"{API}/{file_id}", headers=headers, timeout=60))
        status = body.get("status", "unknown")
        percent = body.get("percent_done", 0)
        pages_done = body.get("num_pages_completed")
        formats = body.get("formats", {})
        marker = (status, percent, pages_done, tuple(sorted(formats.items())))
        if marker != last:
            format_note = ", ".join(
                f"{name}={value}" for name, value in sorted(formats.items()))
            print(
                f"{status}: {percent}% ({pages_done or 0}/669 pages)"
                + (f"; {format_note}" if format_note else ""))
            last = marker
        manifest["status"] = status
        # The account identifiers of the status are not kept.
        manifest["last_status"] = {k: v for k, v in body.items() if k not in ("app_id", "group_id")}
        manifest["status_checked_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "mathpix_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        failed = sorted(name for name, state in formats.items()
                        if state == "error")
        if failed:
            manifest["conversion_failures"] = {
                name: "Mathpix reported conversion status error"
                for name in failed
            }
            (out / "mathpix_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            raise RuntimeError(
                "Mathpix OCR completed, but requested conversion(s) failed: "
                + ", ".join(failed))
        if status == "completed":
            return manifest
        if status == "error":
            raise RuntimeError(f"Mathpix OCR failed: {body}")
        time.sleep(interval)


def redact(path: Path) -> tuple[int, int]:
    """Replace every e-mail address and remove every link to a page crop in the
    strings of a JSON file, and count both.

    The strings are changed after parsing, so an escape such as \\n before an
    address stays intact. Mathpix writes compact JSON, so the file is written
    back in that form and is otherwise unchanged.
    """
    count = images = 0

    def clean(value):
        nonlocal count, images
        if isinstance(value, str):
            value, n = EMAIL.subn(REMOVED, value)
            value, m = IMAGE.subn("", value)
            count, images = count + n, images + m
            return value
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    data = clean(json.loads(path.read_text(encoding="utf-8")))
    if count or images:
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    return count, images


def download(manifest: dict, out: Path, headers: dict[str, str]) -> dict:
    """Download the line-level OCR and hash it."""
    file_id = manifest["mathpix_file_id"]
    outputs = {"lines.json": out / "mathpix_lines.json"}
    records = dict(manifest.get("outputs", {}))
    for extension, path in outputs.items():
        format_state = manifest.get("last_status", {}).get("formats", {}).get(
            extension)
        if format_state == "error":
            manifest.setdefault("conversion_failures", {})[extension] = (
                "Mathpix reported conversion status error")
            print(f"skipped {extension}: Mathpix conversion failed")
            continue
        # A retry after a late conversion must not redownload already retained
        # artifacts. Recompute their hashes so the manifest remains complete.
        if path.exists() and path.stat().st_size:
            records[extension] = {
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "content_type": records.get(extension, {}).get("content_type"),
            }
            print(f"retained {extension}: {path} ({path.stat().st_size:,} bytes)")
            manifest["outputs"] = records
            (out / "mathpix_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            continue
        response = requests.get(
            f"{API}/{file_id}.{extension}", headers=headers,
            timeout=600, stream=True)
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = {"content_type": response.headers.get("content-type")}
            raise RuntimeError(
                f"download {extension} failed with HTTP {response.status_code}: "
                f"{json.dumps(detail, sort_keys=True)[:1000]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
        removed, images = redact(path)
        records[extension] = {
            "path": str(path.relative_to(HERE)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content_type": response.headers.get("content-type"),
            "email_addresses_removed": removed,
            "image_links_removed": images,
        }
        print(f"downloaded {extension}: {path} ({path.stat().st_size:,} bytes)")
        manifest["outputs"] = records
        (out / "mathpix_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    manifest["outputs"] = records
    manifest["downloaded_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "mathpix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--env", type=Path, default=Path.home() / ".env")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()
    if not any((args.submit, args.poll, args.download, args.all)):
        parser.error("choose --submit, --poll, --download, or --all")
    key = mathpix_key(args.env)
    headers = {"app_key": key}
    manifest = None
    if args.submit or args.all:
        manifest = submit(args.source.resolve(), args.out.resolve(), headers)
    if args.poll or args.all:
        manifest = manifest or load_manifest(args.out.resolve())
        manifest = poll(
            manifest, args.out.resolve(), headers, max(2, args.poll_interval))
    if args.download or args.all:
        manifest = manifest or load_manifest(args.out.resolve())
        if manifest.get("status") != "completed":
            raise RuntimeError("OCR is not completed; run --poll first")
        download(manifest, args.out.resolve(), headers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
