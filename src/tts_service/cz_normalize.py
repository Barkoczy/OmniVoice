"""Czech text normalization for text-to-speech.

OmniVoice does not normalize input text, and its README explicitly warns that
digits should be converted to words before synthesis. This module expands the
most common written forms into spoken Czech:

- cardinal numbers (incl. thousands separators) with correct Czech grammar,
- decimal numbers ("3,14" -> "tři celá jedna čtyři"),
- percentages, currencies and common units with number agreement,
- frequent abbreviations ("atd." -> "a tak dále").

Known limitations (kept simple on purpose, documented for honesty):
- ordinals ("1." -> "první") are NOT expanded (ambiguous without context),
- dates are not parsed,
- the dot is never treated as a thousands separator (too ambiguous),
- grammatical case is approximated using the standard 1 / 2-4 / 5+ rule.
"""

from __future__ import annotations

import re

__all__ = ["normalize_text", "number_to_words"]

# --- Building blocks ---------------------------------------------------------

_UNITS = ["nula", "jedna", "dva", "tři", "čtyři", "pět",
          "šest", "sedm", "osm", "devět"]
_TEENS = {
    10: "deset", 11: "jedenáct", 12: "dvanáct", 13: "třináct", 14: "čtrnáct",
    15: "patnáct", 16: "šestnáct", 17: "sedmnáct", 18: "osmnáct", 19: "devatenáct",
}
_TENS = {
    2: "dvacet", 3: "třicet", 4: "čtyřicet", 5: "padesát",
    6: "šedesát", 7: "sedmdesát", 8: "osmdesát", 9: "devadesát",
}
_HUNDREDS = {
    1: "sto", 2: "dvě stě", 3: "tři sta", 4: "čtyři sta", 5: "pět set",
    6: "šest set", 7: "sedm set", 8: "osm set", 9: "devět set",
}

# scale value, (singular, 2-4 form, 5+ form), grammatical gender of the noun
_SCALES = [
    (10 ** 12, ("bilion", "biliony", "bilionů"), "m"),
    (10 ** 9, ("miliarda", "miliardy", "miliard"), "f"),
    (10 ** 6, ("milion", "miliony", "milionů"), "m"),
    (10 ** 3, ("tisíc", "tisíce", "tisíc"), "m"),
]


def _below_thousand(n: int) -> str:
    """Spell an integer in the range 1..999."""
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_UNITS[rem])
        elif rem < 20:
            parts.append(_TEENS[rem])
        else:
            t, u = divmod(rem, 10)
            parts.append(_TENS[t] + (" " + _UNITS[u] if u else ""))
    return " ".join(parts)


def _gender_fix(words: str, gender: str) -> str:
    """Adjust the words for 'one' and 'two' to match the noun gender."""
    if gender == "f":  # feminine: jedna, dvě
        words = re.sub(r"\bdva\b", "dvě", words)
    elif gender == "m":  # masculine: jeden, dva
        words = re.sub(r"\bjedna\b", "jeden", words)
    elif gender == "n":  # neuter: jedno, dvě
        words = re.sub(r"\bjedna\b", "jedno", words)
        words = re.sub(r"\bdva\b", "dvě", words)
    return words


def _plural_form(count: int, forms: tuple) -> str:
    """Pick the Czech noun form for a count: 1 / 2-4 / 5+ (with teen exception)."""
    if count % 100 in (11, 12, 13, 14):
        return forms[2]
    last = count % 10
    if last == 1:
        return forms[0]
    if last in (2, 3, 4):
        return forms[1]
    return forms[2]


def number_to_words(n: int) -> str:
    """Convert a non-negative-or-negative integer to spoken Czech."""
    if n == 0:
        return "nula"
    if n < 0:
        return "mínus " + number_to_words(-n)

    parts = []
    for scale_val, forms, gender in _SCALES:
        if n >= scale_val:
            count, n = divmod(n, scale_val)
            if count == 1:
                parts.append(forms[0])
            else:
                spoken = _gender_fix(_below_thousand(count), gender)
                parts.append(spoken + " " + _plural_form(count, forms))
    if n > 0:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def _count_noun(n: int, forms: tuple, gender: str) -> str:
    """Spell a count together with its noun, with grammar agreement.

    e.g. _count_noun(1, ("koruna","koruny","korun"), "f") -> "jedna koruna"
         _count_noun(2, ...) -> "dvě koruny"; _count_noun(5, ...) -> "pět korun"
    """
    spoken = _gender_fix(number_to_words(n), gender)
    return spoken + " " + _plural_form(n, forms)


# --- Numeric token helpers ---------------------------------------------------

# A number with optional space/NBSP/thin-space thousands grouping and optional
# decimal comma. The dot is intentionally NOT accepted as a thousands separator.
_NUM = r"\d{1,3}(?:[   ]\d{3})+(?:,\d+)?|\d+(?:,\d+)?"
_SEP = re.compile(r"[   ]")


def _strip_sep(s: str) -> str:
    return _SEP.sub("", s)


def _num_token_to_words(token: str) -> str:
    """Convert a raw numeric token (maybe grouped / decimal) to words."""
    token = token.strip()
    if "," in token:
        int_part, dec_part = token.split(",", 1)
    else:
        int_part, dec_part = token, None
    int_part = _strip_sep(int_part) or "0"
    words = number_to_words(int(int_part))
    if dec_part:
        dec_part = _strip_sep(dec_part)
        spoken_dec = " ".join(_UNITS[int(d)] for d in dec_part)
        words = f"{words} celá {spoken_dec}"
    return words


def _amount_with_noun(token: str, forms: tuple, gender: str) -> str:
    """Number + a counted noun; uses agreement for integers, genitive for decimals."""
    if "," in token:  # decimal: fall back to genitive plural (5+ form)
        return _num_token_to_words(token) + " " + forms[2]
    return _count_noun(int(_strip_sep(token)), forms, gender)


# --- Units / currency / percent ----------------------------------------------

# Mapping of unit token -> (forms, gender). Order matters: longer tokens first.
_UNITS_MAP = [
    ("km/h", (("kilometr za hodinu", "kilometry za hodinu", "kilometrů za hodinu"), "m")),
    ("km", (("kilometr", "kilometry", "kilometrů"), "m")),
    ("cm", (("centimetr", "centimetry", "centimetrů"), "m")),
    ("mm", (("milimetr", "milimetry", "milimetrů"), "m")),
    ("kg", (("kilogram", "kilogramy", "kilogramů"), "m")),
    ("ml", (("mililitr", "mililitry", "mililitrů"), "m")),
    ("°C", (("stupeň Celsia", "stupně Celsia", "stupňů Celsia"), "m")),
    ("°", (("stupeň", "stupně", "stupňů"), "m")),
    ("m", (("metr", "metry", "metrů"), "m")),
    ("g", (("gram", "gramy", "gramů"), "m")),
    ("l", (("litr", "litry", "litrů"), "m")),
]

_CURRENCY_MAP = [
    ("Kč", (("koruna", "koruny", "korun"), "f")),
    ("CZK", (("koruna", "koruny", "korun"), "f")),
    ("€", (("euro", "eura", "eur"), "n")),
    ("EUR", (("euro", "eura", "eur"), "n")),
    ("$", (("dolar", "dolary", "dolarů"), "m")),
    ("USD", (("dolar", "dolary", "dolarů"), "m")),
]

_PERCENT_FORMS = ("procento", "procenta", "procent")  # neuter


def _build_unit_pattern(pairs):
    keys = sorted((re.escape(k) for k, _ in pairs), key=len, reverse=True)
    return re.compile(rf"({_NUM})\s*(" + "|".join(keys) + r")(?![\wáčďéěíňóřšťúůýž])")


_RE_PERCENT = re.compile(rf"({_NUM})\s*%")
_RE_CURRENCY = _build_unit_pattern(_CURRENCY_MAP)
_RE_UNIT = _build_unit_pattern(_UNITS_MAP)
_RE_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_RE_NUM = re.compile(_NUM)

# Gender agreement for a few very common gendered nouns that are often written
# out in full. Only the numeral 'one'/'two' is adjusted; the noun and its
# grammatical case are left untouched (so "2 hodinami" -> "dvě hodinami").
_GENDERED_NOUNS = [
    ("f", r"(?:hodin|minut|sekund|vteřin|korun)\w*"),
    ("n", r"eur(?:[ao])?\w*"),
]
_RE_GENDERED = [
    (re.compile(rf"({_NUM})(\s+)(?={pat}\b)"), gender)
    for gender, pat in _GENDERED_NOUNS
]

_HOUR_FORMS = ("hodina", "hodiny", "hodin")
_MIN_FORMS = ("minuta", "minuty", "minut")


def _lookup(token_key, table):
    for k, v in table:
        if k == token_key:
            return v
    return None


# --- Abbreviations -----------------------------------------------------------

# Multi-token abbreviations (contain spaces/dots) handled first, longest first.
_MULTI_ABBREV = [
    (re.compile(r"př\.\s*n\.\s*l\.", re.IGNORECASE), "před naším letopočtem"),
    (re.compile(r"n\.\s*l\.", re.IGNORECASE), "našeho letopočtu"),
    (re.compile(r"a\s*s\.\b", re.IGNORECASE), "akciová společnost"),
    (re.compile(r"s\.\s*r\.\s*o\.", re.IGNORECASE), "es er o"),
]

_ABBREV = {
    "atd.": "a tak dále",
    "apod.": "a podobně",
    "aj.": "a jiné",
    "tj.": "to jest",
    "tzn.": "to znamená",
    "tzv.": "takzvaný",
    "např.": "například",
    "mj.": "mimo jiné",
    "popř.": "popřípadě",
    "příp.": "případně",
    "resp.": "respektive",
    "č.": "číslo",
    "čís.": "číslo",
    "str.": "strana",
    "obr.": "obrázek",
    "tab.": "tabulka",
    "kap.": "kapitola",
    "roč.": "ročník",
    "stol.": "století",
    "mld.": "miliard",
    "mil.": "milionů",
    "tis.": "tisíc",
    "hod.": "hodin",
    "min.": "minut",
    "sek.": "sekund",
    "ks": "kusů",
    "pozn.": "poznámka",
    "viz": "viz",
}


def _build_abbrev_pattern(keys):
    esc = sorted((re.escape(k) for k in keys), key=len, reverse=True)
    return re.compile(r"(?<![\wáčďéěíňóřšťúůýž])(" + "|".join(esc) + r")", re.IGNORECASE)


_RE_ABBREV = _build_abbrev_pattern(_ABBREV.keys())
_ABBREV_LOWER = {k.lower(): v for k, v in _ABBREV.items()}

_SYMBOLS = [
    (re.compile(r"\s*&\s*"), " a "),
    (re.compile(r"(?<=\d)\s*[×x]\s*(?=\d)"), " krát "),
    (re.compile(r"(?<=\d)\s*\+\s*(?=\d)"), " plus "),
    (re.compile(r"§"), "paragraf "),
    (re.compile(r"@"), " zavináč "),
]


# --- Public entry point ------------------------------------------------------

def normalize_text(text: str, expand_abbrev: bool = True, expand_time: bool = True) -> str:
    """Normalize Czech text for TTS: numbers, units, currency, abbreviations."""
    if not text:
        return text

    if expand_abbrev:
        for pattern, repl in _MULTI_ABBREV:
            text = pattern.sub(repl, text)
        text = _RE_ABBREV.sub(lambda m: _ABBREV_LOWER[m.group(1).lower()], text)

    if expand_time:
        def _time(m):
            h, mnt = int(m.group(1)), int(m.group(2))
            out = _count_noun(h, _HOUR_FORMS, "f")
            if mnt:
                out += " " + _count_noun(mnt, _MIN_FORMS, "f")
            return out
        text = _RE_TIME.sub(_time, text)

    text = _RE_PERCENT.sub(
        lambda m: _amount_with_noun(m.group(1), _PERCENT_FORMS, "n"), text)
    text = _RE_CURRENCY.sub(
        lambda m: _amount_with_noun(m.group(1), *_lookup(m.group(2), _CURRENCY_MAP)), text)
    text = _RE_UNIT.sub(
        lambda m: _amount_with_noun(m.group(1), *_lookup(m.group(2), _UNITS_MAP)), text)
    for regex, gender in _RE_GENDERED:
        text = regex.sub(
            lambda m, g=gender: _gender_fix(_num_token_to_words(m.group(1)), g) + m.group(2),
            text)
    text = _RE_NUM.sub(lambda m: _num_token_to_words(m.group(0)), text)

    for pattern, repl in _SYMBOLS:
        text = pattern.sub(repl, text)

    # collapse whitespace introduced by expansions
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        "Mám 3 jablka a 21 hrušek.",
        "Cena je 1 234,50 Kč a sleva 15 %.",
        "Teplota je 21 °C, vlhkost 55 %.",
        "Viz str. 12, kap. 3, atd.",
        "Schůzka je v 14:30, trvá 2 hodiny.",
        "Vzdálenost 2 km, hmotnost 5 kg, π = 3,14.",
        "Rok 2026 byl př. n. l. nemožný.",
        "Bylo to 1000000 korun, tedy 2 miliony dohromady.",
    ]
    for s in samples:
        print(f"{s}\n  -> {normalize_text(s)}\n")
