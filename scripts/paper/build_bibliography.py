#!/usr/bin/env python3
"""Write paper/Bibliography.bib from bibliographic APIs, not from memory.

The identifier record, paper/bibliography.json, gives each citation key
of paper/PNAS_Paper.tex an identifier: a DOI, an ADS bibcode, an arXiv id, an
ACL Anthology id, an NTRS id or an ISBN. The script fetches each record and
writes the entry from the fields that the record returns:

    Crossref      https://api.crossref.org/works/<doi>
    NASA ADS      https://api.adsabs.harvard.edu (token: the ADS= line of ~/.env)
    arXiv         http://export.arxiv.org/api/query?id_list=<id>
    ACL Anthology https://aclanthology.org/<id>.bib
    NTRS          https://ntrs.nasa.gov/api/citations/<id>
    DNB           https://services.dnb.de/sru/dnb (German National Library)
    Open Library  https://openlibrary.org/books/<edition>.json
    doi.org       https://doi.org/api/handles/<doi> (to reject DOI aliases)

A field that no API gives correctly is set in the record together with its
source, and the script checks it against the API record where one exists. A
fetch that disagrees with the record stops the build.

Only keys cited in the paper (\\cite and \\citeS, multi-key cites included)
are written. The output has no date, so two runs on the same API data give
the same file.

    python3 scripts/paper/build_bibliography.py           # write the .bib
    python3 scripts/paper/build_bibliography.py --check   # fail on drift
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = "build_bibliography/1.0 (Babylonian_Dating paper)"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
MARC = "{http://www.loc.gov/MARC21/slim}"

FIELDS = ["author", "editor", "title", "booktitle", "journal", "series",
          "volume", "number", "pages", "publisher", "address", "institution",
          "type", "howpublished", "year", "isbn", "doi", "eprint",
          "archiveprefix", "url", "note"]

CROSSREF_TYPES = {"journal-article": "article",
                  "proceedings-article": "inproceedings",
                  "book-chapter": "incollection", "monograph": "book",
                  "book": "book", "edited-book": "book"}

ACCENTS = {"\u0301": "'", "\u0300": "`", "\u0308": '"', "\u0302": "^",
           "\u0303": "~", "\u0327": "c", "\u030c": "v", "\u0304": "="}

TEXT = {"\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
        "\u201c": "``", "\u201d": "''", "\u00a0": " ", "\u2212": "$-$",
        "&": "\\&", "%": "\\%", "#": "\\#"}


# --------------------------------------------------------------- fetching

def fetch(url: str, headers: dict | None = None, tries: int = 4) -> bytes:
    """GET a URL with retries; no personal data is sent."""
    request = urllib.request.Request(url, headers={"User-Agent": AGENT,
                                                   **(headers or {})})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_json(url: str, headers: dict | None = None) -> dict:
    return json.loads(fetch(url, headers))


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, "/")
    return fetch_json(url)["message"]


def doi_aliases(doi: str) -> list[str]:
    """Return the DOI that doi.org says this DOI is an alias of, if any."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, "/")
    values = fetch_json(url).get("values", [])
    return [v["data"]["value"] for v in values if v["type"] == "HS_ALIAS"]


def ads_token() -> str:
    """Read the ADS token from ~/.env, as the earlier script read its key."""
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith("ADS="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no ADS= line in ~/.env")


def ads(bibcode: str, token: str) -> dict:
    query = urllib.parse.urlencode({
        "q": f'bibcode:"{bibcode}"',
        "fl": "bibcode,title,author,year,volume,issue,page,doi,identifier"})
    url = "https://api.adsabs.harvard.edu/v1/search/query?" + query
    docs = fetch_json(url, {"Authorization": "Bearer " + token})
    docs = docs["response"]["docs"]
    if len(docs) != 1:
        raise SystemExit(f"ADS has no single record for {bibcode}")
    return docs[0]


def arxiv(ids: list[str]) -> dict[str, dict]:
    """Fetch arXiv records in one request: title, authors, DOI."""
    query = urllib.parse.urlencode({"id_list": ",".join(ids),
                                    "max_results": len(ids)})
    root = ET.fromstring(fetch("http://export.arxiv.org/api/query?" + query))
    found = {}
    for entry in root.findall(ATOM + "entry"):
        ident = entry.find(ATOM + "id").text.rsplit("/abs/", 1)[-1]
        ident = re.sub(r"v\d+$", "", ident)
        doi = entry.find(ARXIV + "doi")
        found[ident] = {
            "title": " ".join(entry.find(ATOM + "title").text.split()),
            "authors": [a.find(ATOM + "name").text
                        for a in entry.findall(ATOM + "author")],
            "doi": doi.text if doi is not None else None}
    return found


def acl(anthology_id: str) -> dict:
    """Fields of the ACL Anthology BibTeX record."""
    text = fetch(f"https://aclanthology.org/{anthology_id}.bib").decode()
    return {name.lower(): value for name, value in
            re.findall(r'(\w+)\s*=\s*"([^"]*)"', text)}


def ntrs(identifier: str) -> dict:
    return fetch_json(f"https://ntrs.nasa.gov/api/citations/{identifier}",
                      {"Accept": "application/json"})


def dnb(isbn: str) -> dict:
    """MARC fields of the German National Library record for an ISBN."""
    query = urllib.parse.urlencode({
        "version": "1.1", "operation": "searchRetrieve",
        "query": "num=" + isbn, "recordSchema": "MARC21-xml",
        "maximumRecords": "5"})
    root = ET.fromstring(fetch("https://services.dnb.de/sru/dnb?" + query))
    record = next(root.iter(MARC + "record"), None)
    if record is None:
        raise SystemExit(f"DNB has no record for ISBN {isbn}")
    fields: dict[str, str] = {}
    for field in record.findall(MARC + "datafield"):
        for sub in field.findall(MARC + "subfield"):
            name = field.get("tag") + sub.get("code")
            value = unicodedata.normalize("NFC", sub.text or "")
            fields[name] = (fields.get(name, "") + " " + value).strip()
    return fields


def openlibrary(edition: str) -> dict:
    return fetch_json(f"https://openlibrary.org/books/{edition}.json")


# ------------------------------------------------------------- formatting

def tex(text: str) -> str:
    """Plain Unicode to BibTeX: accents as macros, dashes as hyphens."""
    out = []
    for char in unicodedata.normalize("NFD", text):
        if char in ACCENTS and out:
            base = out.pop()
            out.append("{\\%s%s}" % (ACCENTS[char], base))
        else:
            out.append(TEXT.get(char, char))
    result = unicodedata.normalize("NFC", "".join(out))
    if not result.isascii():
        raise SystemExit(f"cannot write {text!r} in ASCII")
    return result


def plain(text: str) -> str:
    """Strip markup and space from an API string."""
    text = html.unescape(re.sub(r"<[^>]+>", "", text or ""))
    return " ".join(text.split())


def comparable(text: str) -> str:
    """Lower-case letters and digits only, for checking two records agree."""
    text = re.sub(r"\\[\"'`^~=cv]\{?(\w)\}?", r"\1", text)
    text = re.sub(r"[{}$]", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def protect(title: str, words: list[str]) -> str:
    """Brace words that pnas-new.bst would otherwise put in lower case."""
    for word in words:
        pattern = r"(?<![\w{])" + re.escape(word) + r"(?![\w}])"
        title, count = re.subn(pattern, "{" + word + "}", title)
        if not count:
            raise SystemExit(f"cannot protect {word!r} in {title!r}")
    return title


def given_name(given: str) -> str:
    """Put a full stop after single-letter initials: 'Paula J' -> 'Paula J.'."""
    return " ".join(part + "." if re.fullmatch(r"[A-Z]", part) else part
                    for part in given.split())


def person_list(people: list[dict], overrides: dict) -> str:
    names = []
    for person in people:
        family = person.get("family") or person.get("name", "")
        given = person.get("given", "")
        if family in overrides:
            family, given = overrides[family]["family"], overrides[family]["given"]
        names.append(f"{family}, {given_name(given)}" if given else family)
    return tex(" and ".join(names))


def year_of(work: dict) -> str:
    for name in ("published-print", "issued"):
        parts = (work.get(name) or {}).get("date-parts") or [[None]]
        if parts[0][0]:
            return str(parts[0][0])
    raise SystemExit(f"no year for {work.get('DOI')}")


def full_title(work: dict) -> str:
    title = plain((work.get("title") or [""])[0])
    subtitle = plain((work.get("subtitle") or [""])[0])
    if subtitle and comparable(subtitle) not in comparable(title):
        title = f"{title}: {subtitle}"
    return title


def pages(text: str) -> str:
    return re.sub(r"\s*[-\u2013]+\s*", "--", text)


# --------------------------------------------------------------- building

def from_crossref(key: str, item: dict, work: dict) -> tuple[str, dict]:
    kind = CROSSREF_TYPES.get(work["type"])
    if kind is None:
        raise SystemExit(f"{key}: Crossref type {work['type']} is not mapped")
    overrides = item.get("names", {})
    fields = {"title": tex(full_title(work)), "year": year_of(work),
              "doi": item["doi"]}
    if work.get("author"):
        fields["author"] = person_list(work["author"], overrides)
    if kind in ("article", "inproceedings", "incollection"):
        container = plain((work.get("container-title") or [""])[0])
        fields["journal" if kind == "article" else "booktitle"] = tex(container)
    if kind == "article":
        for name, source in (("volume", "volume"), ("number", "issue")):
            if work.get(source):
                fields[name] = pages(work[source])
    page = work.get("page") or work.get("article-number")
    if page and kind != "book":
        fields["pages"] = pages(page)
    if kind != "article":
        fields["publisher"] = tex(plain(work.get("publisher", "")))
    if kind == "book" and work.get("editor"):
        fields["editor"] = person_list(work["editor"], overrides)
    return kind, fields


def check_title(key: str, declared: str, found: str) -> None:
    if not comparable(found).startswith(comparable(declared)):
        raise SystemExit(f"{key}: title {declared!r} does not match {found!r}")


def check_names(key: str, declared: str, text: str) -> None:
    for name in declared.split(" and "):
        family = comparable(name.split(",")[0])
        if family.split()[-1] not in comparable(text):
            raise SystemExit(f"{key}: {family} is not in {text!r}")


def build(key: str, item: dict, token: str, papers: dict) -> tuple[str, dict]:
    kind = item.get("type")
    fields: dict[str, str] = {}
    declared = item.get("set", {})

    if "doi" in item:
        aliases = doi_aliases(item["doi"])
        if aliases:
            raise SystemExit(f"{key}: doi {item['doi']} is an alias of {aliases}")
        work = crossref(item["doi"])
        kind, fields = from_crossref(key, item, work)
        if "title" in declared:
            check_title(key, declared["title"], full_title(work))
        if "book_doi" in item:
            book = crossref(item["book_doi"])
            fields["booktitle"] = tex(full_title(book))
            fields["editor"] = person_list(book.get("editor", []), {})
        for name in item.get("drop", {}):
            fields.pop(name, None)

    if "ads" in item:
        record = ads(item["ads"], token)
        if fields.get("volume") and record.get("volume") != fields["volume"]:
            raise SystemExit(f"{key}: ADS volume {record.get('volume')}")
        first = fields.get("pages", "").split("--")[0]
        if first and comparable(record["page"][0]) != comparable(first):
            raise SystemExit(f"{key}: ADS page {record['page'][0]}")
        if record["year"] != fields["year"]:
            raise SystemExit(f"{key}: ADS year {record['year']}")
        if item.get("title_from") == "ads":
            fields["title"] = tex(plain(record["title"][0]))

    if "acl" in item:
        record = acl(item["acl"])
        if comparable(record["doi"]) != comparable(fields["doi"]):
            raise SystemExit(f"{key}: ACL DOI {record['doi']}")
        fields["pages"] = record["pages"]
        fields["address"] = record["address"]
        fields["url"] = record["url"]

    if "arxiv" in item:
        record = papers[item["arxiv"]]
        if record["doi"] and comparable(record["doi"]) != comparable(fields["doi"]):
            raise SystemExit(f"{key}: arXiv DOI {record['doi']}")
        ours, theirs = (set(comparable(t).split()) for t in
                        (fields["title"], record["title"]))
        if len(ours & theirs) < 0.75 * len(ours):
            raise SystemExit(f"{key}: arXiv title {record['title']!r}")
        for override in item.get("names", {}).values():
            if override["check"] not in record["authors"] and not any(
                    override["check"] in a for a in record["authors"]):
                raise SystemExit(f"{key}: {override['check']} not on arXiv")
        fields["eprint"] = item["arxiv"]
        fields["archiveprefix"] = "arXiv"

    if "ntrs" in item:
        record = ntrs(item["ntrs"])
        check_title(key, declared["title"], record["title"])
        people = sorted(record["authorAffiliations"],
                        key=lambda a: a.get("sequence", 0))
        names = []
        for person in people:
            given, family = person["meta"]["author"]["name"].rsplit(" ", 1)
            names.append({"family": family, "given": given})
        number = [n for n in record["otherReportNumbers"]
                  if n.startswith("NASA/TP")]
        fields.update({
            "author": person_list(names, {}),
            "institution": tex(record["publications"][0]["publisher"] + ", "
                               + record["center"]["name"]),
            "type": "NASA Technical Publication",
            "number": number[0],
            "year": record["publications"][0]["publicationDate"][:4],
            "url": f"https://ntrs.nasa.gov/citations/{item['ntrs']}"})

    if "dnb" in item:
        record = dnb(item["dnb"]["isbn"])
        year = record.get("264c", record.get("260c", ""))
        series = record.get("490v", "")
        statement = " ".join(record.get(t, "") for t in ("245c", "100a", "700a"))
        where = record.get("264a", record.get("260a", ""))
        target = {**fields, **declared}
        if target["year"] not in year:
            raise SystemExit(f"{key}: DNB year {year}")
        if "number" in target and target["number"] not in series:
            raise SystemExit(f"{key}: DNB series {series}")
        if item["dnb"]["place"] not in where:
            raise SystemExit(f"{key}: DNB place {where}")
        if "245p" in record and comparable(record["245p"]) not in comparable(
                target["title"]):
            raise SystemExit(f"{key}: DNB part title {record['245p']}")
        for name in ("author", "editor"):
            if name in target:
                check_names(key, target[name], statement)

    if "openlibrary" in item:
        record = openlibrary(item["openlibrary"]["edition"])
        target = {**fields, **declared}
        if record["publish_date"] != target["year"]:
            raise SystemExit(f"{key}: Open Library year {record['publish_date']}")
        if not any(target["number"] in s for s in record.get("series", [])):
            raise SystemExit(f"{key}: Open Library series {record.get('series')}")
        if item["openlibrary"]["place"] not in record.get("publish_places", []):
            raise SystemExit(f"{key}: Open Library place")
        check_title(key, record["title"], target["title"])
        check_names(key, target["author"], record.get("by_statement", ""))
        if "publisher" in declared and record["publishers"] != [
                declared["publisher"]]:
            raise SystemExit(f"{key}: Open Library publisher")

    if "crossref_reference" in item:
        spec = item["crossref_reference"]
        refs = crossref(spec["doi"]).get("reference", [])
        ref = next((r for r in refs if r.get("key") == spec["key"]), None)
        if ref is None:
            raise SystemExit(f"{key}: reference {spec['key']} not in Crossref")
        if (comparable(ref.get("journal-title", "")) != comparable(declared["journal"])
                or ref.get("first-page") != declared["pages"].split("--")[0]
                or not declared["year"].startswith(ref.get("year", "?"))):
            raise SystemExit(f"{key}: Crossref reference {ref}")
        check_names(key, declared["author"].split(" and ")[0], ref["author"])

    if "web" in item:
        page = fetch(item["web"]["url"]).decode("utf-8", "replace")
        for text in item["web"]["expect"]:
            if text not in page:
                raise SystemExit(f"{key}: {text!r} is not on {item['web']['url']}")

    if "file" in item:
        spec = item["file"]
        path = ROOT / spec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise SystemExit(f"{key}: checksum of {path} changed")
        with zipfile.ZipFile(path) as archive:
            portal = archive.read(spec["portal"]).decode()
            metadata = json.loads(archive.read(spec["metadata"]))
        for text in spec["expect"]:
            if text not in portal:
                raise SystemExit(f"{key}: {text!r} is not in {spec['portal']}")
        if metadata["UTC-timestamp"][:4] != declared["year"]:
            raise SystemExit(f"{key}: release year {metadata['UTC-timestamp']}")

    fields.update(declared)
    if "isbn" in item:
        fields["isbn"] = item["isbn"]
    if item.get("protect"):
        fields["title"] = protect(fields["title"], item["protect"])
    if not kind:
        raise SystemExit(f"{key}: no entry type")
    return kind, fields


def render(key: str, kind: str, fields: dict) -> str:
    names = [name for name in FIELDS if fields.get(name)]
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise SystemExit(f"{key}: unknown fields {unknown}")
    width = max(len(name) for name in names)
    lines = [f"@{kind}{{{key},"]
    lines += [f"  {name.ljust(width)} = {{{fields[name]}}}," for name in names]
    lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines + ["}"])


def cited_keys(tex_path: Path) -> list[str]:
    """Keys of every \\cite-like command, comments removed."""
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", line)
                     for line in tex_path.read_text().splitlines())
    keys = []
    for group in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}",
                            text):
        keys += [k.strip() for k in group.split(",") if k.strip()]
    return sorted(set(keys), key=str.lower)


def bibliography(tex_path: Path, record_path: Path) -> str:
    record = json.loads(record_path.read_text())
    keys = cited_keys(tex_path)
    missing = [k for k in keys if k not in record]
    if missing:
        raise SystemExit(f"cited but not in {record_path.name}: {missing}")
    unused = [k for k in record if not k.startswith("_") and k not in keys]
    if unused:
        print(f"  not cited, left out: {', '.join(unused)}")
    token = ads_token()
    papers = arxiv(sorted({record[k]["arxiv"] for k in keys
                           if "arxiv" in record[k]}))
    blocks = ["% Generated by scripts/paper/build_bibliography.py from "
              "paper/bibliography.json.",
              "% Do not edit by hand: change the record and run the script.",
              ""]
    for key in keys:
        kind, fields = build(key, record[key], token, papers)
        print(f"  {key:32s} {kind:13s} {fields['year']:8s} "
              f"{fields['title'][:50]}")
        blocks.append(render(key, kind, fields) + "\n")
        time.sleep(0.2)
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="fail if the .bib differs from the API records")
    parser.add_argument("--tex", type=Path,
                        default=ROOT / "paper" / "PNAS_Paper.tex")
    parser.add_argument("--record", type=Path,
                        default=ROOT / "paper" / "bibliography.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "paper" / "Bibliography.bib")
    arguments = parser.parse_args()

    fresh = bibliography(arguments.tex, arguments.record)
    if arguments.check:
        current = arguments.out.read_text() if arguments.out.exists() else ""
        if current == fresh:
            print("bibliography matches the API records")
            return 0
        print("bibliography differs from the API records:")
        sys.stdout.writelines(difflib.unified_diff(
            current.splitlines(True), fresh.splitlines(True),
            str(arguments.out), "from APIs"))
        return 1
    arguments.out.write_text(fresh)
    print(f"{fresh.count('@')} entries -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
