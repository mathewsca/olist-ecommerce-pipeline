"""
FabricaIA - Cleaning Helpers

Reusable, pure (Series/DataFrame in -> Series/DataFrame out) building blocks
for cleaning the Olist tables: text and string sanitization, city / state /
zip code (CEP) standardization, product category handling, numeric and date
typing and null treatment.

StagingTransformer composes these helpers table by table; the notebooks call
them one at a time on small examples so a newcomer can see each rule in
isolation. Every helper that changes data accepts an optional DataQualityLog
and records what it found (rule, rows affected, action, examples).
"""

import html
import re
import unicodedata
from difflib import get_close_matches
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import numpy as np
import pandas as pd

from src.etl.data_quality import DataQualityLog, sample_changes, sample_values

# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------

BR_STATES = frozenset(
    "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()
)

UF_TO_REGION: Dict[str, str] = {
    **{uf: "Norte" for uf in ("AC", "AP", "AM", "PA", "RO", "RR", "TO")},
    **{uf: "Nordeste" for uf in ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")},
    **{uf: "Centro-Oeste" for uf in ("DF", "GO", "MT", "MS")},
    **{uf: "Sudeste" for uf in ("ES", "MG", "RJ", "SP")},
    **{uf: "Sul" for uf in ("PR", "RS", "SC")},
}

# First-5-digits CEP ranges published by Correios, per state.
UF_ZIP_RANGES: List[Tuple[int, int, str]] = [
    (1000, 19999, "SP"), (20000, 28999, "RJ"), (29000, 29999, "ES"), (30000, 39999, "MG"),
    (40000, 48999, "BA"), (49000, 49999, "SE"), (50000, 56999, "PE"), (57000, 57999, "AL"),
    (58000, 58999, "PB"), (59000, 59999, "RN"), (60000, 63999, "CE"), (64000, 64999, "PI"),
    (65000, 65999, "MA"), (66000, 68899, "PA"), (68900, 68999, "AP"), (69000, 69299, "AM"),
    (69300, 69399, "RR"), (69400, 69899, "AM"), (69900, 69999, "AC"), (70000, 72799, "DF"),
    (72800, 72999, "GO"), (73000, 73699, "DF"), (73700, 76799, "GO"), (76800, 76999, "RO"),
    (77000, 77999, "TO"), (78000, 78899, "MT"), (78900, 78999, "RO"), (79000, 79999, "MS"),
    (80000, 87999, "PR"), (88000, 89999, "SC"), (90000, 99999, "RS"),
]

# Rough bounding box of Brazil (including islands), used to reject bad coordinates.
BRAZIL_LAT_RANGE = (-33.8, 5.4)
BRAZIL_LNG_RANGE = (-74.0, -34.7)

ORDER_STATUSES = frozenset(
    {"created", "approved", "invoiced", "processing", "shipped", "delivered", "unavailable", "canceled"}
)

# The Kaggle translation file misses these two categories.
CATEGORY_TRANSLATION_FALLBACK: Dict[str, str] = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}

# Macro groups so the dashboard can read 73 categories as ~12 business areas.
CATEGORY_GROUPS: Dict[str, List[str]] = {
    "Casa e Decoração": [
        "cama_mesa_banho", "moveis_decoracao", "utilidades_domesticas", "moveis_escritorio", "moveis_sala",
        "casa_conforto", "casa_conforto_2", "moveis_cozinha_area_de_servico_jantar_e_jardim",
        "moveis_quarto", "moveis_colchao_e_estofado", "la_cuisine", "artigos_de_natal", "flores",
    ],
    "Eletrodomésticos": [
        "eletrodomesticos", "eletrodomesticos_2", "eletroportateis", "climatizacao",
        "portateis_casa_forno_e_cafe", "portateis_cozinha_e_preparadores_de_alimentos",
    ],
    "Informática e Eletrônicos": [
        "informatica_acessorios", "eletronicos", "pcs", "pc_gamer", "audio", "tablets_impressao_imagem",
        "consoles_games", "cine_foto",
    ],
    "Telefonia": ["telefonia", "telefonia_fixa"],
    "Moda e Acessórios": [
        "fashion_bolsas_e_acessorios", "fashion_calcados", "fashion_roupa_masculina",
        "fashion_roupa_feminina", "fashion_underwear_e_moda_praia", "fashion_esporte",
        "fashion_roupa_infanto_juvenil", "malas_acessorios", "relogios_presentes",
    ],
    "Beleza e Saúde": ["beleza_saude", "perfumaria", "fraldas_higiene"],
    "Esporte e Lazer": ["esporte_lazer", "instrumentos_musicais", "cool_stuff"],
    "Bebês e Brinquedos": ["bebes", "brinquedos"],
    "Alimentos e Bebidas": ["alimentos_bebidas", "alimentos", "bebidas"],
    "Automotivo": ["automotivo"],
    "Construção e Ferramentas": [
        "ferramentas_jardim", "construcao_ferramentas_construcao", "casa_construcao",
        "construcao_ferramentas_seguranca", "construcao_ferramentas_jardim",
        "construcao_ferramentas_iluminacao", "construcao_ferramentas_ferramentas",
        "sinalizacao_e_seguranca",
    ],
    "Livros, Mídia e Papelaria": [
        "livros_interesse_geral", "livros_tecnicos", "livros_importados", "dvds_blu_ray", "musica",
        "cds_dvds_musicais", "papelaria", "artes", "artes_e_artesanato",
    ],
    "Pet Shop": ["pet_shop"],
    "Negócios e Serviços": [
        "market_place", "agro_industria_e_comercio", "industria_comercio_e_negocios", "seguros_e_servicos",
        "artigos_de_festas",
    ],
}
_CATEGORY_TO_GROUP: Dict[str, str] = {c: g for g, cats in CATEGORY_GROUPS.items() for c in cats}

_APOSTROPHES = str.maketrans({"´": "'", "`": "'", "’": "'", "‘": "'", "′": "'"})
_CITY_SEPARATORS = re.compile(r"\s*[/\\,;|]\s*|\s+-\s+")
_CITY_ALLOWED = re.compile(r"[^a-z0-9 '\-.]")
# \x80 (not \u0080): same codepoint, but pandas' pyarrow-backed .str.contains() (used in
# clean_free_text) hands this pattern to RE2, which rejects the \u escape form.
_MOJIBAKE = re.compile(r"[ÃÂ][\x80-¿]|â€")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTISPACE = re.compile(r"\s+")

PT_STOPWORDS = frozenset(
    """a o as os um uma uns umas de do da dos das em no na nos nas por para com sem e ou mas que se ao aos
    foi era ser ter tem eu me meu minha meus minhas voce vc ele ela eles elas sim ja so muito muita mais
    menos bem mal pra pro esta estou isso isto essa esse este como quando onde tudo todo toda mesmo mesma
    ate depois antes ainda porque pois entao aqui la seu sua seus suas nem the produto pedido""".split()
)


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------


def strip_accents(value):
    """Remove diacritics: 'São José' -> 'Sao Jose'. Non-strings are returned unchanged."""
    if not isinstance(value, str):
        return value
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _map_unique(series: pd.Series, func: Callable) -> pd.Series:
    """Apply func once per distinct value (fast on 1M-row columns with few distinct values)."""
    uniques = series.dropna().unique()
    lookup = {value: func(value) for value in uniques}
    return series.map(lookup).where(series.notna(), other=pd.NA)


def _mask(values) -> pd.Series:
    """Coerce a (possibly NA-containing) boolean result to a plain bool Series usable for indexing."""
    return pd.Series(values).fillna(False).astype(bool)


def fix_mojibake(value):
    """
    Repair text that was UTF-8 but decoded as latin-1 ('SÃ£o Paulo' -> 'São Paulo').
    Only touches strings that contain the tell-tale 'Ã'/'Â' + control-range pattern;
    anything that cannot be round-tripped is returned untouched.
    """
    if not isinstance(value, str) or not _MOJIBAKE.search(value):
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """Collapse runs of whitespace (including tabs/newlines) into one space and trim the ends."""
    return series.astype("string").str.replace(_MULTISPACE, " ", regex=True).str.strip()


# ----------------------------------------------------------------------
# Free text (review titles / messages)
# ----------------------------------------------------------------------


def clean_free_text(
    series: pd.Series,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
) -> pd.Series:
    """
    Sanitize customer-written text while keeping it readable (case and accents are preserved):
      1. repair mojibake, 2. drop control characters, 3. turn line breaks/tabs into spaces,
      4. collapse repeated spaces and trim, 5. turn empty strings into NULL.
    """
    before = series.copy()
    total = len(series)

    def _clean(value: str) -> Optional[str]:
        text = fix_mojibake(value)
        text = _CONTROL_CHARS.sub(" ", text)
        text = _MULTISPACE.sub(" ", text).strip()
        return text or None

    cleaned = _map_unique(series, _clean).astype("string")
    if dq is not None:
        had_break = _mask(series.astype("string").str.contains(r"[\r\n\t]", regex=True, na=False))
        dq.add(table, column, "line_breaks_and_tabs", int(had_break.sum()), total, "fixed",
               "Quebras de linha e tabs dentro do texto viram espaço.", "low",
               sample_changes(before[had_break], cleaned[had_break], 2))
        edge_space = _mask(series.astype("string").str.contains(r"^\s|\s$", regex=True, na=False)) & ~had_break
        dq.add(table, column, "edge_and_double_spaces", int(edge_space.sum()), total, "fixed",
               "Espaços nas bordas ou duplicados são removidos.", "info")
        blank = _mask(series.notna() & cleaned.isna())
        dq.add(table, column, "blank_text_to_null", int(blank.sum()), total, "nullified",
               "Texto vazio ou só com espaços vira NULL.", "low")
        moji = _mask(series.astype("string").str.contains(_MOJIBAKE, na=False))
        dq.add(table, column, "mojibake", int(moji.sum()), total, "fixed",
               "Texto com codificação quebrada (Ã£ em vez de ã) é reconvertido.", "medium",
               sample_changes(before[moji], cleaned[moji], 2))
    return cleaned


def text_features(series: pd.Series) -> pd.DataFrame:
    """
    Descriptive flags for cleaned free text: length, word count, all-caps ("shouting")
    and low-information comments ("ok", ".", "10", ...), useful to filter NLP inputs.
    """
    text = series.astype("string")
    length = text.str.len().astype("Int64")
    words = text.str.count(r"\S+").astype("Int64")
    letters = text.str.replace(r"[^A-Za-zÀ-ÿ]", "", regex=True)
    # .astype("boolean") on both sides: the length comparison comes back as pandas'
    # nullable "boolean" dtype while the string comparison comes back as pyarrow's
    # "bool[pyarrow]" - mixing the two in `&` raises "boolean value of NA is ambiguous".
    is_upper = (letters.str.len() > 5).astype("boolean") & (letters == letters.str.upper()).astype("boolean")
    low_info = text.notna() & ((length <= 3) | text.str.fullmatch(r"[\W\d_]+").fillna(False))
    return pd.DataFrame(
        {
            "comment_length": length,
            "comment_word_count": words,
            "is_shouting": is_upper.fillna(False).astype(bool),
            "is_low_information": low_info.fillna(False).astype(bool),
        }
    )


def normalize_for_nlp(series: pd.Series) -> pd.Series:
    """Lowercase, no accents, no URLs, no repeated letters/punctuation ('otimooo!!!' -> 'otimoo!')."""

    def _norm(value: str) -> str:
        text = _URL.sub(" ", strip_accents(value).lower())
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return _MULTISPACE.sub(" ", text).strip()

    return _map_unique(series, _norm).astype("string")


def tokenize_pt(text: str, stopwords: frozenset = PT_STOPWORDS, min_len: int = 3) -> List[str]:
    """Split normalized text into word tokens, dropping stopwords and very short tokens."""
    words = re.findall(r"[a-z]+", strip_accents(text).lower())
    return [w for w in words if len(w) >= min_len and w not in stopwords]


# ----------------------------------------------------------------------
# State (UF), region and zip code (CEP)
# ----------------------------------------------------------------------


def standardize_state(
    series: pd.Series, table: str = "", column: str = "", dq: Optional[DataQualityLog] = None
) -> pd.Series:
    """Uppercase and trim the UF; anything that is not one of the 27 Brazilian UFs becomes NULL."""
    upper = series.astype("string").str.strip().str.upper()
    valid = upper.isin(BR_STATES)
    out = upper.where(valid, other=pd.NA)
    if dq is not None:
        total = len(series)
        changed = _mask(series.notna() & (series.astype("string") != upper) & valid)
        dq.add(table, column, "state_case_and_spaces", int(changed.sum()), total, "fixed",
               "UF em minúsculas ou com espaços é padronizada em MAIÚSCULAS.", "low",
               sample_changes(series[changed], upper[changed], 2))
        invalid = _mask(series.notna() & ~valid)
        dq.add(table, column, "invalid_state", int(invalid.sum()), total, "nullified",
               "Valor que não é uma das 27 UFs brasileiras vira NULL.", "high",
               sample_values(series[invalid]))
    return out


def state_to_region(series: pd.Series) -> pd.Series:
    """Map a UF to its IBGE macro-region (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)."""
    return series.map(UF_TO_REGION)


def standardize_zip(
    series: pd.Series, table: str = "", column: str = "", dq: Optional[DataQualityLog] = None
) -> pd.Series:
    """
    Standardize a zip code prefix to a 5-character string.

    Reading the CSV turns '01046' into the integer 1046, so leading zeros are lost.
    Steps: keep digits only, left-pad with zeros up to 5, keep the first 5 digits of a
    full 8-digit CEP ('12345-678'), and turn anything else (or '00000') into NULL.
    """
    original = series.astype("string")

    def _fix(value: str) -> Optional[str]:
        digits = re.sub(r"\D", "", re.sub(r"\.0+$", "", value.strip()))
        if not digits:
            return None
        if len(digits) == 8:  # full CEP '12345-678' -> prefix
            digits = digits[:5]
        if len(digits) > 5:
            return None
        digits = digits.zfill(5)
        return None if digits == "00000" else digits

    out = _map_unique(original, _fix).astype("string")
    if dq is not None:
        total = len(series)
        padded = _mask(original.notna() & out.notna() & original.str.replace(r"\D", "", regex=True).str.len().lt(5))
        dq.add(table, column, "zip_lost_leading_zeros", int(padded.sum()), total, "fixed",
               "CEP lido como número perdeu zeros à esquerda (1046 -> 01046).", "medium",
               sample_changes(original[padded], out[padded], 3))
        invalid = _mask(original.notna() & out.isna())
        dq.add(table, column, "invalid_zip", int(invalid.sum()), total, "nullified",
               "CEP vazio, com mais de 8 dígitos ou '00000' vira NULL.", "high",
               sample_values(original[invalid]))
    return out


def zip_state(zip_code: Optional[str]) -> Optional[str]:
    """Return the UF that owns a 5-digit zip prefix according to the Correios ranges, or None."""
    if not isinstance(zip_code, str) or not zip_code.isdigit():
        return None
    value = int(zip_code)
    for low, high, uf in UF_ZIP_RANGES:
        if low <= value <= high:
            return uf
    return None


def zip_state_mismatch(zip_series: pd.Series, state_series: pd.Series) -> pd.Series:
    """True when the UF implied by the zip range differs from the row's own UF (NA-safe -> False)."""
    expected = _map_unique(zip_series.astype("string"), zip_state)
    known = expected.notna() & state_series.notna()
    return (known & (expected.astype("string") != state_series.astype("string"))).fillna(False).astype(bool)


# ----------------------------------------------------------------------
# City names
# ----------------------------------------------------------------------


def _clean_city_value(raw: str, uf: Optional[str]) -> Optional[str]:
    text = fix_mojibake(raw)
    text = html.unescape(unquote(text))
    # Apostrophe look-alikes must be translated BEFORE accent stripping: NFKD turns the
    # acute accent sign U+00B4 into "space + combining acute", which would leave a stray space.
    text = strip_accents(text.lower().translate(_APOSTROPHES))
    text = _MULTISPACE.sub(" ", text).strip()
    if not text or "@" in text or re.fullmatch(r"[\d\s.\-]+", text):
        return None
    parts = [p.strip() for p in _CITY_SEPARATORS.split(text) if p.strip()]
    text = parts[0] if parts else ""
    text = re.sub(r"\s*\(.*?\)?\s*$", "", text)
    text = re.sub(r"\s*-\s*distrito$", "", text)
    if uf:
        text = re.sub(rf"(?:\s+|\s*-\s*){uf.lower()}$", "", text)
    text = _MULTISPACE.sub(" ", _CITY_ALLOWED.sub("", text)).strip(" -.")
    text = re.sub(r"(?<=[a-z])\d+$", "", text)  # 'maceia³' -> 'maceia' (digit glued to the last letter)
    if not text or text.upper() in BR_STATES:
        return None
    return text


def standardize_city(
    series: pd.Series,
    state: Optional[pd.Series] = None,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
    repair_with_vocabulary: bool = False,
) -> pd.Series:
    """
    Normalize city names to lowercase, accent-free, single-spaced values.

    Handles the noise found in Olist: accents and casing ('São Paulo'), stray
    apostrophes ('d´oeste'), 'city / uf' and 'city, state, brasil' suffixes,
    parenthesised districts, e-mails and numbers typed into the city field
    (turned into NULL so they can be repaired from the CEP later), and broken
    encodings. With `repair_with_vocabulary`, names that had garbage characters (e.g. 'maceia³')
    and are still unknown are matched to the closest city seen in clean rows of the same UF.
    """
    before = series.copy()
    total = len(series)
    uf = (state if state is not None else pd.Series(pd.NA, index=series.index, dtype="string"))
    uf = uf.astype("string").str.upper().fillna("")
    pairs = pd.DataFrame({"city": series.astype("string"), "uf": uf})
    distinct = pairs.dropna(subset=["city"]).drop_duplicates().copy()
    distinct["clean"] = [_clean_city_value(c, u or None) for c, u in zip(distinct["city"], distinct["uf"])]
    out = pairs.merge(distinct, on=["city", "uf"], how="left")["clean"].astype("string")
    out.index = series.index

    if repair_with_vocabulary:
        garbage = _mask(series.astype("string").str.contains(r"[^A-Za-z0-9À-ÿ \-'.]", regex=True, na=False))
        known = pd.DataFrame({"uf": uf, "city": out})[~garbage & out.notna()]
        vocab_by_uf = {u: sorted(set(g["city"])) for u, g in known.groupby("uf")}
        for uf_value, city_value in pairs.assign(clean=out)[garbage & out.notna()][["uf", "clean"]].drop_duplicates().itertuples(index=False):
            candidates = vocab_by_uf.get(uf_value, [])
            if city_value in candidates:
                continue
            match = get_close_matches(city_value, candidates, n=1, cutoff=0.8)
            if match:
                out = out.mask(_mask((out == city_value) & (uf == uf_value)), match[0])

    if dq is not None:
        raw_norm = normalize_whitespace(series).str.lower()
        cosmetic = _mask(series.notna() & out.notna() & (raw_norm != out))
        accents = _mask(series.map(lambda v: isinstance(v, str) and v != strip_accents(v)))
        dq.add(table, column, "city_accents_and_case", int(accents.sum()), total, "fixed",
               "Acentos e maiúsculas removidos para permitir JOIN entre tabelas (são paulo = sao paulo).",
               "medium", sample_changes(before[accents], out[accents], 3))
        structural = cosmetic & ~accents
        dq.add(table, column, "city_noise_removed", int(structural.sum()), total, "fixed",
               "Sufixos de UF/estado/país, parênteses, apóstrofos e símbolos removidos "
               "(sao paulo / sp -> sao paulo).", "medium", sample_changes(before[structural], out[structural], 3))
        invalid = _mask(series.notna() & out.isna())
        dq.add(table, column, "invalid_city_to_null", int(invalid.sum()), total, "nullified",
               "E-mail, número ou só a sigla da UF no campo cidade vira NULL (depois reparado pelo CEP).",
               "high", sample_values(series[invalid]))
    return out


def fill_city_from_zip(
    city: pd.Series,
    zip_series: pd.Series,
    zip_to_city: Dict[str, str],
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
) -> pd.Series:
    """Fill NULL cities with the city registered for the same CEP in the geolocation reference."""
    missing = city.isna() & zip_series.notna()
    fill = zip_series.map(zip_to_city)
    out = city.mask(missing & fill.notna(), fill).astype("string")
    if dq is not None:
        repaired = missing & fill.notna()
        dq.add(table, column, "city_repaired_from_zip", int(repaired.sum()), len(city), "imputed",
               "Cidade nula (ou inválida) preenchida com a cidade do mesmo CEP na geolocation.", "medium",
               sample_changes(city[repaired], out[repaired], 3))
        dq.add(table, column, "city_still_null", int(out.isna().sum()), len(city), "flagged",
               "Cidade que continua nula: o CEP não existe na tabela de geolocation.", "medium")
    return out


def repair_city_with_reference(
    city: pd.Series,
    state: pd.Series,
    zip_series: pd.Series,
    zip_to_city: Dict[str, str],
    cities_by_state: Dict[str, set],
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
) -> pd.Series:
    """
    Replace cities that do not exist in the state (per the geolocation reference) with the city
    registered for the row's CEP: 'sbc' in SP becomes 'sao bernardo do campo'. Rows whose CEP is
    unknown to the reference are left untouched.
    """
    known = pd.Series(
        [(c in cities_by_state.get(u, ())) for c, u in zip(city, state)], index=city.index, dtype="bool"
    )
    reference_city = zip_series.map(zip_to_city)
    unknown = _mask(city.notna() & ~known & reference_city.notna())
    out = city.mask(unknown, reference_city).astype("string")
    if dq is not None:
        dq.add(table, column, "city_unknown_in_state_replaced_by_zip", int(unknown.sum()), len(city), "fixed",
               "Cidade que não existe no estado (ex.: sigla 'sbc') é trocada pela cidade do CEP na geolocation.",
               "medium", sample_changes(city[unknown], out[unknown], 3))
    return out


# ----------------------------------------------------------------------
# Product categories
# ----------------------------------------------------------------------


def standardize_category(
    series: pd.Series, table: str = "", column: str = "", dq: Optional[DataQualityLog] = None
) -> pd.Series:
    """snake_case, accent-free, trimmed category slug; blank -> NULL."""

    def _slug(value: str) -> Optional[str]:
        slug = re.sub(r"[\s\-]+", "_", strip_accents(value).strip().lower())
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        return slug or None

    out = _map_unique(series, _slug).astype("string")
    if dq is not None:
        changed = _mask(series.notna() & (series.astype("string") != out))
        dq.add(table, column, "category_slug_format", int(changed.sum()), len(series), "fixed",
               "Categoria padronizada em snake_case sem acentos/espaços.", "low",
               sample_changes(series[changed], out[changed], 2))
    return out


def translate_category(
    category_pt: pd.Series,
    translation: pd.DataFrame,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
    missing_label: str = "unknown",
) -> pd.Series:
    """
    Translate category slugs to English with the Kaggle translation table plus
    CATEGORY_TRANSLATION_FALLBACK for the categories that table forgot.
    Missing categories get `missing_label`.
    """
    mapping = dict(zip(translation["product_category_name"], translation["product_category_name_english"]))
    fallback_used = category_pt.isin(CATEGORY_TRANSLATION_FALLBACK) & ~category_pt.isin(mapping)
    mapping = {**CATEGORY_TRANSLATION_FALLBACK, **mapping}
    translated = category_pt.map(mapping)
    untranslated = category_pt.notna() & translated.isna()
    out = translated.fillna(missing_label).astype("string")
    if dq is not None:
        total = len(category_pt)
        dq.add(table, column, "category_missing_in_translation_table", int(fallback_used.sum()), total, "fixed",
               "Categorias que existem nos produtos mas faltam na tabela de tradução do Kaggle "
               "(pc_gamer, portateis_cozinha_...) recebem tradução manual, em vez de virar 'unknown'.",
               "medium", sample_values(category_pt[fallback_used]))
        dq.add(table, column, "category_without_translation", int(untranslated.sum()), total, "flagged",
               "Categoria sem tradução mesmo após o fallback.", "medium", sample_values(category_pt[untranslated]))
    return out


def category_group(category_pt: pd.Series) -> pd.Series:
    """Map a Portuguese category slug to its business macro-group ('Sem categoria' when NULL)."""
    grouped = category_pt.map(_CATEGORY_TO_GROUP)
    return grouped.where(~(category_pt.notna() & grouped.isna()), "Outros").fillna("Sem categoria")


# ----------------------------------------------------------------------
# Numbers and dates
# ----------------------------------------------------------------------


def parse_number(series: pd.Series) -> pd.Series:
    """
    Convert text such as 'R$ 1.234,56', '1234,5' or '12.5' to float; anything unparseable is NaN.
    Already-numeric input is returned as float.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip().str.replace(r"(?i)r\$|\s", "", regex=True)
    both = text.str.contains(r"\.", na=False) & text.str.contains(",", na=False)
    text = text.mask(both, text.str.replace(".", "", regex=False)).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def coerce_numeric(
    series: pd.Series,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    min_inclusive: bool = True,
    decimals: Optional[int] = None,
) -> pd.Series:
    """
    Numeric typing + range sanitization.

    Unparseable values and values outside [minimum, maximum] become NaN (nullified and
    logged), so nothing invalid reaches the warehouse; `decimals` rounds money-like columns.
    """
    parsed = parse_number(series)
    total = len(series)
    bad_type = _mask(series.notna() & parsed.isna())
    out = parsed
    if minimum is not None:
        below = out < minimum if min_inclusive else out <= minimum
        below = below.fillna(False)
    else:
        below = pd.Series(False, index=series.index)
    above = (out > maximum).fillna(False) if maximum is not None else pd.Series(False, index=series.index)
    out = out.mask(below | above)
    if decimals is not None:
        out = out.round(decimals)
    if dq is not None:
        dq.add(table, column, "not_numeric", int(bad_type.sum()), total, "nullified",
               "Valor que não pode ser convertido para número vira NULL.", "high", sample_values(series[bad_type]))
        if minimum is not None:
            dq.add(table, column, "below_minimum", int(below.sum()), total, "nullified",
                   f"Valor {'<' if min_inclusive else '<='} {minimum} é fisicamente/comercialmente inválido "
                   "e vira NULL.", "high", sample_values(series[below]))
        if maximum is not None:
            dq.add(table, column, "above_maximum", int(above.sum()), total, "nullified",
                   f"Valor > {maximum} é impossível e vira NULL.", "high", sample_values(series[above]))
    return out


def iqr_outlier_mask(series: pd.Series, k: float = 3.0) -> pd.Series:
    """True for values beyond Q3 + k*IQR (k=1.5 'mild', k=3 'extreme'); NaN-safe."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    q1, q3 = clean.quantile([0.25, 0.75])
    fence = q3 + k * (q3 - q1)
    return (series > fence).fillna(False)


def parse_datetime(
    series: pd.Series,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> pd.Series:
    """
    Parse timestamps (ISO first, then day-first 'dd/mm/yyyy' as a fallback). Unparseable values
    and dates outside [min_date, max_date] become NaT and are logged.
    """
    total = len(series)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
        retry = parsed.isna() & series.notna()
        if retry.any():
            parsed = parsed.where(~retry, pd.to_datetime(series[retry], errors="coerce", dayfirst=True))
    bad_format = _mask(series.notna() & parsed.isna())
    out_of_range = pd.Series(False, index=series.index)
    if min_date is not None:
        out_of_range |= (parsed < pd.Timestamp(min_date)).fillna(False)
    if max_date is not None:
        out_of_range |= (parsed > pd.Timestamp(max_date)).fillna(False)
    result = parsed.mask(out_of_range)
    if dq is not None:
        dq.add(table, column, "unparseable_date", int(bad_format.sum()), total, "nullified",
               "Texto que não é uma data válida vira NaT (NULL).", "high", sample_values(series[bad_format]))
        if min_date is not None or max_date is not None:
            dq.add(table, column, "date_out_of_range", int(out_of_range.sum()), total, "nullified",
                   f"Data fora da janela plausível {min_date or '...'} a {max_date or '...'} vira NULL.",
                   "high", sample_values(parsed[out_of_range].dt.strftime("%Y-%m-%d")))
    return result


# ----------------------------------------------------------------------
# Null treatment
# ----------------------------------------------------------------------


def fill_missing(
    series: pd.Series,
    value,
    table: str = "",
    column: str = "",
    dq: Optional[DataQualityLog] = None,
    rule: str = "null_filled_with_constant",
    description: str = "",
) -> pd.Series:
    """Replace NULLs with a constant (sentinel) and log how many were filled."""
    missing = series.isna()
    out = series.fillna(value)
    if dq is not None:
        dq.add(table, column, rule, int(missing.sum()), len(series), "imputed",
               description or f"Nulos preenchidos com o valor constante {value!r}.", "medium")
    return out


def impute_by_group_median(
    df: pd.DataFrame,
    value_columns: Sequence[str],
    group_column: str,
    table: str = "",
    dq: Optional[DataQualityLog] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fill NULLs of `value_columns` with the median of the row's group (e.g. product category),
    falling back to the global median. Returns the frame plus a boolean Series marking rows
    where at least one value was imputed, so the imputation stays visible downstream.
    """
    out = df.copy()
    imputed_any = pd.Series(False, index=df.index)
    for column in value_columns:
        missing = out[column].isna()
        if not missing.any():
            if dq is not None:
                dq.add(table, column, "null_imputed_group_median", 0, len(out), "imputed",
                       "Nulos preenchidos com a mediana da categoria (fallback: mediana global).", "medium")
            continue
        group_median = out.groupby(group_column)[column].transform("median")
        out[column] = out[column].fillna(group_median).fillna(out[column].median())
        imputed_any |= missing
        if dq is not None:
            dq.add(table, column, "null_imputed_group_median", int(missing.sum()), len(out), "imputed",
                   "Nulos preenchidos com a mediana da categoria (fallback: mediana global).", "medium")
    return out, imputed_any


def filter_gps_outliers(
    lat: pd.Series, lng: pd.Series, group: pd.Series, max_offset_deg: float = 1.0
) -> pd.Series:
    """
    True for coordinates that are inside Brazil and within `max_offset_deg` degrees of the median
    point of their group (zip code). Drops GPS outliers without needing a reference geocoder.
    """
    in_brazil = lat.between(*BRAZIL_LAT_RANGE) & lng.between(*BRAZIL_LNG_RANGE)
    med_lat = lat.where(in_brazil).groupby(group).transform("median")
    med_lng = lng.where(in_brazil).groupby(group).transform("median")
    near = ((lat - med_lat).abs() <= max_offset_deg) & ((lng - med_lng).abs() <= max_offset_deg)
    return (in_brazil & near.fillna(False)).astype(bool)


def report_duplicates(
    df: pd.DataFrame,
    subset: Sequence[str],
    table: str,
    dq: Optional[DataQualityLog] = None,
    keep: str = "first",
    description: str = "",
) -> pd.DataFrame:
    """Drop duplicates on the natural key and log how many rows were dropped."""
    duplicated = df.duplicated(subset=list(subset), keep=keep)
    if dq is not None:
        dq.add(table, "+".join(subset), "duplicate_key", int(duplicated.sum()), len(df), "dropped",
               description or f"Linhas repetidas para a chave ({', '.join(subset)}) foram removidas.",
               "high" if duplicated.sum() else "info")
    return df.loc[~duplicated].copy()


def as_string_id(series: pd.Series) -> pd.Series:
    """Trim and lowercase an identifier column, keeping it as a string dtype."""
    return series.astype("string").str.strip().str.lower()
