"""
FabricaIA ETL - Tests for the reusable cleaning helpers (src/etl/cleaning.py).

Each test class documents one family of rules found in the real Olist data.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl import cleaning as cl
from src.etl.data_quality import DataQualityLog

BACKSLASH = chr(92)


def _findings(dq: DataQualityLog) -> dict:
    return {issue.rule: issue.rows_affected for issue in dq.issues}


class TestTextBasics:
    def test_strip_accents_handles_composed_and_decomposed_forms(self):
        assert cl.strip_accents("São José") == "Sao Jose"
        assert cl.strip_accents("são paulo") == "sao paulo"  # 'a' + combining tilde
        assert cl.strip_accents(None) is None
        assert cl.strip_accents(42) == 42

    def test_fix_mojibake_repairs_utf8_read_as_latin1_but_spares_real_text(self):
        assert cl.fix_mojibake("SÃ£o Paulo") == "São Paulo"
        assert cl.fix_mojibake("NÃO recebi o produto") == "NÃO recebi o produto"  # legitimate capital Ã
        assert cl.fix_mojibake("sem problema") == "sem problema"

    def test_normalize_whitespace(self):
        out = cl.normalize_whitespace(pd.Series(["  a   b\t\nc ", None]))
        assert out.iloc[0] == "a b c"
        assert pd.isna(out.iloc[1])


class TestFreeText:
    def test_clean_free_text_collapses_breaks_and_nullifies_blank(self):
        dq = DataQualityLog()
        raw = pd.Series(["  Entrega\r\nrápida   demais  ", "   ", "Ótimo", None])
        out = cl.clean_free_text(raw, "reviews", "message", dq)

        assert out.iloc[0] == "Entrega rápida demais"
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == "Ótimo"  # case and accents are preserved
        found = _findings(dq)
        assert found["line_breaks_and_tabs"] == 1
        assert found["blank_text_to_null"] == 1

    def test_clean_free_text_repairs_mojibake(self):
        out = cl.clean_free_text(pd.Series(["chegou rÃ¡pido"]))
        assert out.iloc[0] == "chegou rápido"

    def test_text_features_flags_low_information_and_shouting(self):
        text = pd.Series(["ok", ".", "10", "PRODUTO CHEGOU QUEBRADO", "Gostei muito do produto", None])
        feats = cl.text_features(text)

        assert feats["is_low_information"].tolist() == [True, True, True, False, False, False]
        assert feats["is_shouting"].tolist() == [False, False, False, True, False, False]
        assert feats["comment_word_count"].iloc[4] == 4
        assert feats["comment_length"].iloc[0] == 2

    def test_normalize_for_nlp_lowercases_removes_accents_urls_and_repeats(self):
        out = cl.normalize_for_nlp(pd.Series(["ÓTIMOOOO!!! veja https://x.com/a", None]))
        assert out.iloc[0] == "otimoo!! veja"
        assert pd.isna(out.iloc[1])

    def test_tokenize_pt_drops_stopwords_and_short_tokens(self):
        # 'nao' is deliberately NOT a stopword: it is the strongest signal in complaints ("nao chegou").
        assert cl.tokenize_pt("O produto não chegou e a entrega atrasou") == ["nao", "chegou", "entrega", "atrasou"]


class TestStateAndRegion:
    def test_standardize_state_uppercases_and_rejects_invalid(self):
        dq = DataQualityLog()
        out = cl.standardize_state(pd.Series(["sp", " rj ", "XX", "São Paulo", None]), "customers", "state", dq)

        assert out.iloc[:2].tolist() == ["SP", "RJ"]
        assert out.iloc[2:].isna().all()
        found = _findings(dq)
        assert found["state_case_and_spaces"] == 2
        assert found["invalid_state"] == 2

    def test_state_to_region(self):
        regions = cl.state_to_region(pd.Series(["SP", "BA", "RS", "AM", "DF"]))
        assert regions.tolist() == ["Sudeste", "Nordeste", "Sul", "Norte", "Centro-Oeste"]


class TestZipCode:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (1046, "01046"),          # leading zero lost when the CSV was read as a number
            ("01046", "01046"),
            ("1046.0", "01046"),      # float artifact
            ("12345-678", "12345"),   # full 8-digit CEP -> 5-digit prefix
            ("12345678", "12345"),
            ("123456", None),         # neither a prefix nor a full CEP
            ("abc", None),
            ("00000", None),
        ],
    )
    def test_standardize_zip(self, raw, expected):
        out = cl.standardize_zip(pd.Series([raw]))
        assert (out.iloc[0] == expected) if expected else pd.isna(out.iloc[0])

    def test_standardize_zip_logs_padded_and_invalid(self):
        dq = DataQualityLog()
        cl.standardize_zip(pd.Series([1046, 99999, "abc"]), "customers", "zip", dq)
        found = _findings(dq)
        assert found["zip_lost_leading_zeros"] == 1
        assert found["invalid_zip"] == 1

    def test_zip_state_uses_correios_ranges(self):
        assert cl.zip_state("01046") == "SP"
        assert cl.zip_state("20040") == "RJ"
        assert cl.zip_state("72800") == "GO"
        assert cl.zip_state("70040") == "DF"
        assert cl.zip_state("90010") == "RS"
        assert cl.zip_state("00100") is None
        assert cl.zip_state(None) is None

    def test_zip_state_mismatch_is_false_when_unknown(self):
        zips = pd.Series(["01046", "01046", None, "20040"])
        states = pd.Series(["SP", "RJ", "SP", None])
        assert cl.zip_state_mismatch(zips, states).tolist() == [False, True, False, False]


class TestCityStandardization:
    @pytest.mark.parametrize(
        "raw, uf, expected",
        [
            ("São Paulo", "SP", "sao paulo"),
            ("são paulo", "SP", "sao paulo"),
            ("sao  jose dos pinhais", "PR", "sao jose dos pinhais"),
            ("santa barbara d´oeste", "SP", "santa barbara d'oeste"),   # acute accent sign as apostrophe
            ("sao paulo / sao paulo", "SP", "sao paulo"),
            ("auriflama/sp", "SP", "auriflama"),
            ("lages - sc", "SC", "lages"),
            ("andira-pr", "PR", "andira"),
            ("sao paulo sp", "SP", "sao paulo"),
            ("novo hamburgo, rio grande do sul, brasil", "RS", "novo hamburgo"),
            ("rio de janeiro " + BACKSLASH + "rio de janeiro", "RJ", "rio de janeiro"),
            ("arraial d'ajuda (porto seguro)", "BA", "arraial d'ajuda"),
            ("sao joao do pau d%26apos%3balho", "SP", "sao joao do pau d'alho"),
            ("SBC/SP", "SP", "sbc"),
            ("riacho fundo 2", "DF", "riacho fundo 2"),
        ],
    )
    def test_noise_is_removed(self, raw, uf, expected):
        out = cl.standardize_city(pd.Series([raw]), pd.Series([uf]))
        assert out.iloc[0] == expected

    @pytest.mark.parametrize("raw", ["vendas@creditparts.com.br", "04482255", "sp", "SP / SP", ""])
    def test_invalid_values_become_null(self, raw):
        out = cl.standardize_city(pd.Series([raw]), pd.Series(["SP"]))
        assert pd.isna(out.iloc[0])

    def test_null_stays_null_and_logging_counts_findings(self):
        dq = DataQualityLog()
        raw = pd.Series(["São Paulo", "lages - sc", "04482255", None])
        out = cl.standardize_city(raw, pd.Series(["SP", "SC", "SP", "SP"]), "sellers", "city", dq)

        assert pd.isna(out.iloc[3])
        found = _findings(dq)
        assert found["city_accents_and_case"] == 1
        assert found["city_noise_removed"] == 1
        assert found["invalid_city_to_null"] == 1

    def test_vocabulary_repair_matches_garbage_names_within_the_same_state(self):
        raw = pd.Series(["maceio", "maceio", "maceia" + chr(0x00B3), "sa" + chr(0x00A3) + "o paulo", "sao paulo"])
        uf = pd.Series(["AL", "AL", "AL", "SP", "SP"])
        out = cl.standardize_city(raw, uf, repair_with_vocabulary=True)
        assert out.tolist() == ["maceio", "maceio", "maceio", "sao paulo", "sao paulo"]

    def test_fill_city_from_zip_only_fills_nulls(self):
        city = pd.Series(["campinas", pd.NA, pd.NA], dtype="string")
        zips = pd.Series(["13010", "04482", "99999"], dtype="string")
        dq = DataQualityLog()
        out = cl.fill_city_from_zip(city, zips, {"04482": "sao paulo", "13010": "outra"}, "sellers", "city", dq)

        assert out.iloc[0] == "campinas"
        assert out.iloc[1] == "sao paulo"
        assert pd.isna(out.iloc[2])
        assert _findings(dq) == {"city_repaired_from_zip": 1, "city_still_null": 1}

    def test_repair_city_with_reference_replaces_names_unknown_in_the_state(self):
        city = pd.Series(["sbc", "campinas", "sao pauo"], dtype="string")
        state = pd.Series(["SP", "SP", "SP"], dtype="string")
        zips = pd.Series(["09710", "13010", "01310"], dtype="string")
        reference = {"09710": "sao bernardo do campo", "13010": "campinas", "01310": "sao paulo"}
        out = cl.repair_city_with_reference(city, state, zips, reference, {"SP": {"campinas", "sao paulo", "sao bernardo do campo"}})

        assert out.tolist() == ["sao bernardo do campo", "campinas", "sao paulo"]


class TestCategories:
    def test_standardize_category_makes_snake_case_without_accents(self):
        out = cl.standardize_category(pd.Series(["Cama Mesa-Banho", " Informática ", "", None]))
        assert out.iloc[0] == "cama_mesa_banho"
        assert out.iloc[1] == "informatica"
        assert out.iloc[2:].isna().all()

    def test_translate_category_uses_fallback_for_categories_missing_in_the_kaggle_file(self):
        translation = pd.DataFrame(
            {"product_category_name": ["moveis_decoracao"], "product_category_name_english": ["furniture_decor"]}
        )
        dq = DataQualityLog()
        categories = pd.Series(["moveis_decoracao", "pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos", "inventada", None])
        out = cl.translate_category(categories, translation, "products", "cat_en", dq)

        assert out.tolist() == [
            "furniture_decor", "pc_gamer", "portable_kitchen_food_preparers", "unknown", "unknown",
        ]
        found = _findings(dq)
        assert found["category_missing_in_translation_table"] == 2
        assert found["category_without_translation"] == 1

    def test_category_group_maps_known_unknown_and_missing(self):
        out = cl.category_group(pd.Series(["cama_mesa_banho", "telefonia", "categoria_nova", None]))
        assert out.tolist() == ["Casa e Decoração", "Telefonia", "Outros", "Sem categoria"]

    def test_every_grouped_category_belongs_to_a_single_group(self):
        all_categories = [c for cats in cl.CATEGORY_GROUPS.values() for c in cats]
        assert len(all_categories) == len(set(all_categories))


class TestNumbers:
    def test_parse_number_handles_brazilian_formats(self):
        out = cl.parse_number(pd.Series(["R$ 1.234,56", "1234,5", "12.5", "abc", None, " 7 "]))
        assert out.iloc[:3].tolist() == [1234.56, 1234.5, 12.5]
        assert out.iloc[3:5].isna().all()
        assert out.iloc[5] == 7

    def test_coerce_numeric_nullifies_out_of_range_and_logs(self):
        dq = DataQualityLog()
        out = cl.coerce_numeric(
            pd.Series(["10.129", "x", "-5", "0", "1000"]), "items", "price", dq,
            minimum=0, min_inclusive=False, maximum=500, decimals=2,
        )
        assert out.iloc[0] == pytest.approx(10.13)
        assert out.iloc[1:].isna().all()
        found = _findings(dq)
        assert found["not_numeric"] == 1
        assert found["below_minimum"] == 2  # -5 and 0
        assert found["above_maximum"] == 1

    def test_iqr_outlier_mask(self):
        values = pd.Series([10, 11, 12, 13, 14, 15, 500, np.nan])
        assert cl.iqr_outlier_mask(values, k=3.0).tolist() == [False] * 6 + [True, False]
        assert not cl.iqr_outlier_mask(pd.Series([np.nan, np.nan])).any()


class TestDates:
    def test_parse_datetime_accepts_iso_then_day_first(self):
        out = cl.parse_datetime(pd.Series(["2018-03-04 10:00:00", "04/03/2018", "not a date", None]))
        assert out.iloc[0] == pd.Timestamp("2018-03-04 10:00:00")
        assert out.iloc[1] == pd.Timestamp("2018-03-04")
        assert out.iloc[2:].isna().all()

    def test_parse_datetime_nullifies_dates_outside_the_plausible_window(self):
        dq = DataQualityLog()
        out = cl.parse_datetime(
            pd.Series(["2018-01-01", "2020-02-05", "1999-01-01"]), "items", "limit", dq,
            min_date="2016-01-01", max_date="2019-12-31",
        )
        assert out.notna().tolist() == [True, False, False]
        assert _findings(dq)["date_out_of_range"] == 2


class TestNullsAndDuplicates:
    def test_fill_missing_uses_constant_and_logs(self):
        dq = DataQualityLog()
        out = cl.fill_missing(pd.Series(["a", None, None]), "unknown", "products", "category", dq)
        assert out.tolist() == ["a", "unknown", "unknown"]
        assert _findings(dq) == {"null_filled_with_constant": 2}

    def test_impute_by_group_median_marks_imputed_rows(self):
        df = pd.DataFrame({"cat": ["a", "a", "a", "b", "c"], "w": [10.0, 30.0, np.nan, np.nan, 5.0]})
        out, imputed = cl.impute_by_group_median(df, ["w"], "cat", "products")

        assert out["w"].tolist()[2] == 20.0            # median of group 'a'
        assert out["w"].tolist()[3] == 10.0            # group 'b' has no value -> global median (10, 30, 5)
        assert imputed.tolist() == [False, False, True, True, False]

    def test_filter_gps_outliers_drops_points_outside_brazil_or_far_from_the_zip_cluster(self):
        lat = pd.Series([-23.5, -23.6, -23.55, 40.0, -10.0])
        lng = pd.Series([-46.6, -46.7, -46.65, -3.0, -46.6])
        zips = pd.Series(["01000"] * 5)
        assert cl.filter_gps_outliers(lat, lng, zips).tolist() == [True, True, True, False, False]

    def test_report_duplicates_drops_and_logs(self):
        dq = DataQualityLog()
        out = cl.report_duplicates(pd.DataFrame({"k": [1, 1, 2]}), ["k"], "t", dq)
        assert len(out) == 2
        assert _findings(dq) == {"duplicate_key": 1}

    def test_as_string_id_trims_and_lowercases(self):
        assert cl.as_string_id(pd.Series([" ABC123 ", None])).iloc[0] == "abc123"
