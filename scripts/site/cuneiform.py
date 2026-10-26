"""Turn the edition's transliteration back into cuneiform signs, for the page in docs/.

The sign list is ORACC's OSL (data/source/osl/osl.asl, CC0). Each sign there
lists its readings and its Unicode glyph. A line of transliteration is split into
its signs, and each reading is looked up in the list. An example line is
"GE₆ 8 USAN GU₄ . ⸢ UD ⸣ [ e MUL₂ - IGI - ša₂ ...".

Square brackets mark broken signs and half brackets damaged ones. A whole number
below 60 is drawn as tens and units, as the Diaries write it. A reading that the
list does not hold stays in Latin letters. scripts/site/build_site_data.py calls
this module.
"""

from __future__ import annotations

import re
from pathlib import Path

OSL = Path(__file__).resolve().parents[2] / "data" / "source" / "osl" / "osl.asl"


def sign_table(path: Path = OSL) -> dict[str, str]:
    """Each reading of the sign list, with the glyph of its sign."""
    table: dict[str, str] = {}
    clash: set[str] = set()
    glyph = None
    values: list[str] = []

    def close():
        if glyph:
            for v in values:
                if v in table and table[v] != glyph:
                    clash.add(v)
                table.setdefault(v, glyph)

    in_form = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@sign "):
            close()
            glyph, values, in_form = None, [], False
            name = line.split(None, 1)[1].strip()
            values.append(name.lower())
        elif line.startswith("@form "):
            in_form = True              # a variant form, whose readings stay with the form
        elif line.startswith("@@"):
            in_form = False
        elif line.startswith("@ucun") and not in_form and glyph is None:
            glyph = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else None
        elif line.startswith("@v\t") and not in_form:
            v = line[3:].strip()
            # A reading marked @v is a qualified one, and it may belong to another sign too.
            if v.endswith("@v"):
                continue
            if v and not v.startswith("%") and ("~" not in v or "(" in v):
                values.append(v)            # a numeral's variant form, such as 9(diš)~a, may stand in
        elif line.startswith("@end sign"):
            close()
            glyph, values, in_form = None, [], False
    close()
    for v in clash:                     # a reading of two signs has no single glyph
        table.pop(v, None)
    return table


# ADART I writes the index of a reading with an accent, á for 2 and à for 3.
ACUTE = dict(zip("áéíúÁÉÍÚ", "aeiuAEIU"))
GRAVE = dict(zip("àèìùÀÈÌÙ", "aeiuAEIU"))


def normalize(reading: str) -> str:
    """Return a reading in the form of the sign list.

    The form is lower case, with h for ḫ. An accent becomes a subscript index,
    so á becomes a₂ and à becomes a₃.
    """
    out, index = "", ""
    for ch in reading:
        if ch in ACUTE:
            out, index = out + ACUTE[ch], "₂"
        elif ch in GRAVE:
            out, index = out + GRAVE[ch], "₃"
        else:
            out += ch
    out = out.replace("ḫ", "h").replace("Ḫ", "H").lower()
    return out + index if index and not re.search(r"[₀-₉]$", out) else out


def _glyph(key: str, table: dict[str, str]) -> str | None:
    """The glyph of a reading, or else of its variant form marked ~a."""
    return table.get(key) or table.get(f"{key}~a")


def _number(n: int, table: dict[str, str]) -> str | None:
    """The glyphs of a number below 60, as tens and then units."""
    tens, units = divmod(n, 10)
    parts = ([f"{tens}(u)"] if tens else []) + ([f"{units}(diš)"] if units else [])
    glyphs = [_glyph(p, table) for p in parts]
    return "".join(glyphs) if parts and all(glyphs) else None


def to_cuneiform(translit: str, table: dict[str, str]) -> list[list]:
    """The signs of one line, as [kind, text, state, reading] items.

    kind is "sign", "latin", "gap" or "space". state is "", "damaged" or "broken", and
    for a gap also "unread" (an illegible sign) or "blank" (an empty space on the clay).
    """
    out: list[list] = []
    broken = damaged = omitted = False
    note: list[str] | None = None
    # Rebuild the words, because the edition puts spaces around its hyphens and dots.
    text = re.sub(r"\s*-\s*", "-", translit or "")
    text = re.sub(r"\s+\.\s+", ".", text)
    # A counted unit such as "3( BARIG )" is one sign, 3(barig).
    text = re.sub(r"(\d)\(\s*([^()\s]+)\s*\)", r"\1(\2)", text)
    # Three dots mark lost signs. One character keeps them apart from the dots
    # that join signs.
    text = text.replace("...", "…")
    for word in text.split():
        before = len(out)
        # A note of the editors in parentheses, such as "(traces)" or "(blank)", is not a sign.
        if note is None and (word.startswith("(") or word.startswith("n(")):
            note = []
        if note is not None:
            note.append(word)
            broken = broken if "[" not in word and "]" not in word else word.rfind("[") > word.rfind("]")
            if ")" in word:
                said, note_first = " ".join(note), note[0]
                note = None
                if note_first.startswith("n(") or any(t in said for t in ("traces", "erasure", "…")):
                    out.append(["gap", "", "unread", said])
                elif "blank" in said:
                    out.append(["gap", "", "blank", said])
            if len(out) > before:
                out.append(["space", "", "", " "])
            continue
        if word == ")":
            continue
        pieces = re.split(r"([-.])", word)
        for piece in pieces:
            if piece in ("-", "."):
                continue
            # Brackets can stand alone or touch a reading.
            for token in re.findall(r"<<|>>|<|>|\[|\]|⸢|⸣|[^\[\]⸢⸣<>]+", piece):
                # Signs in <...> are left out by the scribe and supplied by the editors, so they are
                # not on the clay. Signs in <<...>> are written by mistake, and they are on the clay.
                if token in ("<<", ">>"):
                    continue
                if token in ("<", ">"):
                    omitted = token == "<"
                    continue
                if token == "[":
                    broken = True
                    continue
                if token == "]":
                    broken = False
                    continue
                if token == "⸢":
                    damaged = True
                    continue
                if token == "⸣":
                    damaged = False
                    continue
                state = "broken" if broken else ("damaged" if damaged else "")
                core = token.strip("?!*#")
                if not core or omitted or core in ("+", "×"):
                    continue
                # A number such as "1+" or "+2" has more signs lost beside it.
                core = re.sub(r"^\+(?=\d)|(?<=\d)\+$", "", core)
                if core.startswith("…") or core == "..":
                    out.append(["gap", "", "broken", "…"])
                    continue
                if core.lower() in ("x", "n"):
                    out.append(["gap", "", "unread", core])
                    continue
                if core.isascii() and core.isdigit() and int(core) < 60:
                    g = _number(int(core), table)
                    out.append(["sign", g, state, core] if g else ["latin", core, state, core])
                    continue
                if re.fullmatch(r"\d/\d", core):
                    g = _glyph(f"{core}(diš)", table)
                    out.append(["sign", g, state, core] if g else ["latin", core, state, core])
                    continue
                key = normalize(core)
                g = _glyph(key, table)
                out.append(["sign", g, state, core] if g else ["latin", core, state, core])
        if len(out) > before:
            out.append(["space", "", "", " "])
    while out and out[-1][0] == "space":
        out.pop()
    return out
