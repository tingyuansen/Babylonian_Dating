#!/usr/bin/env python3
"""Fetch transliteration and published translation for every diary of ADART I to III.

The bulk JSON download carries the transliteration but not the line-by-line
English. The rendered pages carry both, so the text is taken from there. This is
the same material the inherited corpus holds, fetched directly rather than
through anyone's intermediate copy, which is the point.

Everything is CC0. The translations are Sachs and Hunger's, as reproduced by the
ADsD edition with its post-publication collations.

Output is one JSONL file:

    tablet, project, designation, lines[{label, translit, translation}], text

Politeness. Modest concurrency and a small delay between requests. This hits a
university server maintained by people doing us a favour by publishing openly.

TLS note. Verification fails against the anaconda CA bundle on this machine, so
the system store is tried first and the fallback is announced rather than silent.

Usage:
    python3 fetch_texts.py                 # all volumes
    python3 fetch_texts.py --limit 20      # smoke test
    python3 fetch_texts.py --projects adsd/adart1
"""

from __future__ import annotations

import argparse
import html
import json
import re
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = "https://oracc.museum.upenn.edu"
SYSTEM_CA = "/etc/ssl/cert.pem"

_ctx = None


def ctx() -> ssl.SSLContext:
    global _ctx
    if _ctx is None:
        for cand in (ssl.create_default_context(cafile=SYSTEM_CA),
                     ssl.create_default_context()):
            try:
                urllib.request.urlopen(ROOT, context=cand, timeout=30).close()
                _ctx = cand
                return _ctx
            except Exception:                               # noqa: BLE001
                continue
        print("! TLS verification unavailable locally, continuing unverified")
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        _ctx = c
    return _ctx


ROW_RE = re.compile(r'<tr[^>]*class="l"[^>]*>(.*?)</tr>', re.S)
CELL_RE = {
    "label": re.compile(r'<span class="xlabel[^"]*">(.*?)</span>', re.S),
    "translit": re.compile(r'<td[^>]*class="tlit"[^>]*>(.*?)</td>', re.S),
    "translation": re.compile(r'<td[^>]*class="t1 xtr"[^>]*>(.*?)</td>', re.S),
}


def strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


ATF_LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")


def parse_atf(raw: str) -> list[dict]:
    """Fallback for tablets ADsD transliterates but has not translated.

    About a fifth of the diaries serve raw ATF rather than a rendered table.
    They carry the transliteration and a lemmatisation line but no English. The
    lemmatisation is kept, since it glosses each word and is most of what a
    translation would give.
    """
    flat = strip_tags(raw)
    # ATF runs from the @obverse marker to the end of the text block.
    i = flat.find("@obverse")
    if i < 0:
        return []
    body = flat[i:]
    out, cur = [], None
    for tok in re.split(r"(?=\s\d+'?\.\s)|(?=#lem:)|(?=@\w+)", body):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("@"):
            continue
        if tok.startswith("#lem:"):
            if cur:
                cur["lemma"] = tok[5:].strip()
            continue
        m = ATF_LINE.match(tok)
        if m:
            cur = {"label": m.group(1), "translit": m.group(2).strip(),
                   "translation": "", "lemma": ""}
            out.append(cur)
    return out


def parse(raw: str) -> list[dict]:
    """One entry per rendered line, using the table structure rather than text.

    ORACC marks each line as a <tr class="l"> holding a label span, a tlit cell
    and a t1 xtr cell. Parsing the table structure prevents adjacent line
    transliteration and translation cells from being merged.
    """
    out = []
    for row in ROW_RE.findall(raw):
        rec = {}
        for key, rx in CELL_RE.items():
            m = rx.search(row)
            rec[key] = strip_tags(m.group(1)) if m else ""
        if rec["translation"] or rec["translit"]:
            # The translation cell repeats the label in parentheses. Drop it.
            rec["translation"] = re.sub(r"^\(\s*[or]\s*\d+'?\s*\)\s*", "",
                                        rec["translation"])
            out.append(rec)
    return out


def fetch_one(project: str, tablet: str, retries: int = 3) -> dict | None:
    url = f"{ROOT}/{project}/{tablet}/html"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, context=ctx(), timeout=90) as r:
                raw = r.read().decode("utf-8", "ignore")
            break
        except Exception:                                   # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    lines = parse(raw)
    source = "rendered"
    if not lines:
        lines = parse_atf(raw)
        source = "atf" if lines else "none"
    return {"tablet": tablet, "project": project, "source": source,
            "n_lines": len(lines), "lines": lines,
            "text": "\n".join(l["translation"] for l in lines if l["translation"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "source" / "adsd")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "source" / "adsd" / "texts.jsonl")
    ap.add_argument("--projects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import zipfile
    targets: dict[str, str] = {}
    master = args.raw / "adsd.zip"
    if master.exists():
        with zipfile.ZipFile(master) as z:
            corpus = json.loads(z.read("adsd/corpus.json"))
            targets.update(corpus.get("proxies", {}))
    if args.projects:
        targets = {k: v for k, v in targets.items() if v in args.projects}
    targets = [(v, k) for k, v in sorted(targets.items()) if v]
    if args.limit:
        targets = targets[:args.limit]
    print(f"{len(targets)} tablets to fetch")

    done, failed = [], []

    def work(pair):
        proj, tid = pair
        rec = fetch_one(proj, tid)
        time.sleep(0.25)
        return rec if rec else {"tablet": tid, "project": proj, "n_lines": 0,
                                "lines": [], "text": "", "error": True}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(work, targets), 1):
            (failed if rec.get("error") or rec["n_lines"] == 0 else done).append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(targets)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for rec in done + failed:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n{len(done)} fetched with text, {len(failed)} empty or failed")
    print(f"written to {args.out}")
    if failed:
        print("empty/failed:", ", ".join(r["tablet"] for r in failed[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
