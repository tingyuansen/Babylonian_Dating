#!/usr/bin/env python3
"""Download the Astronomical Diaries Digital (ADsD) corpus from ORACC.

ADsD is the digital edition of Sachs and Hunger, prepared at the Austrian
Academy of Sciences under Reinhard Pirngruber. ORACC publishes it under CC0
(SI Appendix, section S1). build_corpus.py and fetch_texts.py read these files.

Outputs, in data/source/adsd/:

    adsd.zip            catalogue of 401 diaries. Each entry gives date_bce,
                        ancient_year, museum_no, genre, period and provenience.
                        The text is in the projects of the single volumes.
    adsd-adart1.zip     the ADART I project. Its catalogue lists all 417
                        diaries of ADART I to III, with the 16 that adsd.zip
                        lacks.
    manifest.json       source, licence, time of download, and the size and
                        SHA-256 of each file

The projects of ADART II and III repeat this catalogue. Those of ADART V and VI
hold other genres, which the paper does not use.

The field date_bce is written as `262/1`, one Babylonian year as the two
Julian years that it spans. The field ancient_year gives the year of the
Seleucid Era for most entries, and a regnal year for the others.

For TLS, the script tries the system certificate store first, then the default
one. If both fail, it prints a warning and downloads without verification. The
manifest records for each file whether verification passed.

Usage:
    python3 scripts/source/fetch_adsd.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://oracc.museum.upenn.edu/json"
PARTS = ["adsd", "adsd-adart1"]
SYSTEM_CA = "/etc/ssl/cert.pem"


def get(url: str, dest: Path) -> tuple[int, bool]:
    """Download url to dest. Return the size in bytes and whether TLS was verified."""
    for ctx, verified in ((ssl.create_default_context(cafile=SYSTEM_CA), True),
                          (ssl.create_default_context(), True)):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
                dest.write_bytes(r.read())
            return dest.stat().st_size, verified
        except Exception:                                   # noqa: BLE001
            continue
    # Both verified attempts failed. Download without verification, and say so.
    print(f"    ! TLS verification failed for {url}, retrying unverified")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
        dest.write_bytes(r.read())
    return dest.stat().st_size, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "source" / "adsd")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {"source": "https://oracc.org/adsd",
                "licence": "CC0 1.0",
                "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": {}}

    for part in PARTS:
        dest = args.out / f"{part}.zip"
        print(f"  {part} ...", end=" ", flush=True)
        size, verified = get(f"{BASE}/{part}.zip", dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        manifest["files"][part] = {"bytes": size, "sha256": digest,
                                   "tls_verified": verified}
        print(f"{size/1e6:.1f} MB")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written to {args.out / 'manifest.json'}")
    print("Data is CC0. Cite the project as ADsD, Austrian Academy of Sciences, "
          "led by Reinhard Pirngruber, published via ORACC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
