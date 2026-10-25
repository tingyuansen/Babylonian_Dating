#!/usr/bin/env python3
"""Download the Astronomical Diaries Digital corpus from ORACC, with provenance.

ADsD is the digital edition of Sachs and Hunger prepared at the Austrian Academy
of Sciences under Reinhard Pirngruber, published through ORACC and released
under CC0. Reading it directly preserves catalogue identifiers and removes
intermediate hand-transcription from the provenance chain.

What arrives:

    adsd.zip            catalogue for 401 diaries. Every entry carries
                        date_bce, ancient_year, museum_no, genre, period and
                        provenience. The text itself is proxied out to the
                        per-volume projects.
    adsd-adart1.zip     the ADART I project. Its catalogue lists all 417
                        diaries of ADART I to III, the 16 that adsd.zip
                        lacks included.

The other per-volume projects repeat this catalogue (ADART II and III) or hold
the other genres of ADART V and VI, which the paper does not use.

The catalogue is the important part. `date_bce` is written as `262/1`, which is
the Babylonian year across the two Julian years it spans, which fixes the date
field convention directly from the catalogue.
`ancient_year` gives the era or regnal year, 345 of 401 in the Seleucid Era and
the rest regnal.

TLS NOTE. Verification fails against the anaconda CA bundle on this machine,
not because of anything wrong at ORACC. The script tries the system trust store
first and only falls back to unverified with an explicit warning, so a silent
downgrade cannot happen.

Usage:
    python3 fetch_adsd.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://oracc.museum.upenn.edu/json"
PARTS = ["adsd", "adsd-adart1"]
SYSTEM_CA = "/etc/ssl/cert.pem"


def get(url: str, dest: Path) -> tuple[int, bool]:
    """Fetch url to dest. Returns (bytes, verified)."""
    for ctx, verified in ((ssl.create_default_context(cafile=SYSTEM_CA), True),
                          (ssl.create_default_context(), True)):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
                dest.write_bytes(r.read())
            return dest.stat().st_size, verified
        except Exception:                                   # noqa: BLE001
            continue
    # Certificate verification is broken locally. Say so rather than hide it.
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
