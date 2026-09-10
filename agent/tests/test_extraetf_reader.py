"""Tests for the read-only extraETF export reader.

All fixtures are synthetic, shaped after the anonymized exports posted in
HKUDS/Vibe-Trading#1170. No real portfolio data is used.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.portfolio.extraetf import (
    ExtraEtfFormatError,
    parse_export,
    read_export,
)

HOLDINGS_COLUMNS = [
    "ISIN",
    "Name",
    "Typ",
    "Anzahl",
    "Kaufpreis",
    "Aktueller Kurs",
    "Aktueller Wert",
    "Währung",
    "Wechselkurs",
    "Region",
    "Land",
    "Sektor",
    "Portfolioname",
    "Portfolio ID",
]

TRANSACTIONS_COLUMNS = [
    "Datum",
    "ISIN",
    "Name",
    "Typ",
    "Transaktion",
    "Anzahl",
    "Preis",
    "Gebühren",
    "Steuern",
    "Währung",
    "Wechselkurs",
    "Portfolioname",
    "Portfolio ID",
    "Stückzinsen",
]

#: A real, checksum-valid ISIN, used so the happy path is realistic.
VALID_ISIN = "US0378331005"
#: The anonymized export's placeholder ISIN, which is shaped like an ISIN but has
#: an invalid check digit.
PLACEHOLDER_ISIN = "IE00AAAA0000"


def _holding(
    *,
    isin: str = VALID_ISIN,
    name: str = "Example World ETF",
    instrument_type: str = "ETF",
    anzahl: str = "12,3456",
    kaufpreis: str = "101,234567",
    kurs: str = "112,345",
    wert: str = "1.386,608",
    waehrung: str = "EUR",
    wechselkurs: str = "1,00",
    region: str = "Welt",
    land: str = "",
    sektor: str = "",
    portfolio: str = "Example broker",
    portfolio_id: str = "1000001",
) -> dict[str, str]:
    return {
        "ISIN": isin,
        "Name": name,
        "Typ": instrument_type,
        "Anzahl": anzahl,
        "Kaufpreis": kaufpreis,
        "Aktueller Kurs": kurs,
        "Aktueller Wert": wert,
        "Währung": waehrung,
        "Wechselkurs": wechselkurs,
        "Region": region,
        "Land": land,
        "Sektor": sektor,
        "Portfolioname": portfolio,
        "Portfolio ID": portfolio_id,
    }


def _movement(
    *,
    datum: str = "01.09.2026",
    isin: str = VALID_ISIN,
    name: str = "Example World ETF",
    instrument_type: str = "ETF",
    transaktion: str = "Kauf",
    anzahl: str = "1,2345",
    preis: str = "110,123456",
    gebuehren: str = "0,00",
    steuern: str = "0,00",
    waehrung: str = "EUR",
    wechselkurs: str = "1,00",
    portfolio: str = "Example broker",
    portfolio_id: str = "1000001",
    stueckzinsen: str = "0",
) -> dict[str, str]:
    return {
        "Datum": datum,
        "ISIN": isin,
        "Name": name,
        "Typ": instrument_type,
        "Transaktion": transaktion,
        "Anzahl": anzahl,
        "Preis": preis,
        "Gebühren": gebuehren,
        "Steuern": steuern,
        "Währung": waehrung,
        "Wechselkurs": wechselkurs,
        "Portfolioname": portfolio,
        "Portfolio ID": portfolio_id,
        "Stückzinsen": stueckzinsen,
    }


def _export(columns: list[str], rows: list[dict[str, str]]) -> str:
    lines = [";".join(columns)]
    lines.extend(";".join(row[column] for column in columns) for row in rows)
    return "\n".join(lines) + "\n"


def _holdings_export(*rows: dict[str, str]) -> str:
    return _export(HOLDINGS_COLUMNS, list(rows))


def _transactions_export(*rows: dict[str, str]) -> str:
    return _export(TRANSACTIONS_COLUMNS, list(rows))


def test_holdings_export_normalizes_a_german_locale_position():
    export = parse_export(_holdings_export(_holding()))

    assert export.kind == "holdings"
    (position,) = export.require_positions()
    assert position["instrument_id"] == VALID_ISIN
    assert position["instrument_id_kind"] == "isin"
    assert position["instrument_id_checksum_ok"] is True
    assert position["symbol"] == VALID_ISIN
    assert position["broker"] == "extraetf"
    assert position["quantity"] == pytest.approx(12.3456)
    assert position["cost_price"] == pytest.approx(101.234567)
    assert position["market_price"] == pytest.approx(112.345)
    assert position["source_market_value"] == pytest.approx(1386.608)
    assert position["currency"] == "EUR"
    assert position["price_currency"] == "EUR"
    assert position["asset_type"] == "etf"
    assert position["instrument_type"] == "ETF"
    assert position["portfolio_id"] == "1000001"
    assert position["portfolio_name"] == "Example broker"
    assert position["region"] == "Welt"
    assert position["country"] is None
    assert position["sector"] is None


def test_german_separators_are_not_read_as_us_numbers():
    """``1.386,608`` is 1386.608, not 1.386 and not 1386608."""
    export = parse_export(
        _holdings_export(
            _holding(
                anzahl="12,3456",
                kaufpreis="1.386,608",
                kurs="59.999,99",
                wert="1.386.608,42",
                wechselkurs="1,0855",
            )
        )
    )

    (position,) = export.require_positions()
    assert position["quantity"] == pytest.approx(12.3456)
    assert position["cost_price"] == pytest.approx(1386.608)
    assert position["market_price"] == pytest.approx(59999.99)
    assert position["source_market_value"] == pytest.approx(1386608.42)
    assert position["fx_rate"] == pytest.approx(1.0855)


def test_us_formatted_number_is_refused_rather_than_misread():
    with pytest.raises(ExtraEtfFormatError, match="US-formatted"):
        parse_export(_holdings_export(_holding(kaufpreis="1,234.56")))


@pytest.mark.parametrize("ambiguous", ["1.23", "1234.56", "0.5"])
def test_ambiguous_separator_grouping_is_refused(ambiguous):
    """A period that is not a thousands separator cannot be guessed at."""
    with pytest.raises(ExtraEtfFormatError, match="ambiguous separators"):
        parse_export(_holdings_export(_holding(kurs=ambiguous)))


@pytest.mark.parametrize("unreadable", ["zwei", "10,00 EUR", "--", ","])
def test_unreadable_number_is_refused(unreadable):
    with pytest.raises(ExtraEtfFormatError, match="is not a number|ambiguous|empty"):
        parse_export(_holdings_export(_holding(anzahl=unreadable)))


def test_blank_optional_fields_are_none_rather_than_zero():
    export = parse_export(
        _holdings_export(_holding(kaufpreis="", kurs="", wert="", wechselkurs=""))
    )

    (position,) = export.require_positions()
    assert position["cost_price"] is None
    assert position["market_price"] is None
    assert position["source_market_value"] is None
    assert position["fx_rate"] is None


def test_transactions_export_yields_movements_and_never_positions():
    """A transactions export is a movement history, not a portfolio."""
    export = parse_export(
        _transactions_export(
            _movement(transaktion="Kauf"),
            _movement(transaktion="Verkauf", anzahl="2,5000", preis="19,876543"),
            _movement(
                transaktion="Dividende",
                anzahl="0,5000",
                preis="2,345",
                steuern="0,45",
            ),
            _movement(transaktion="Einbuchung", anzahl="5,0000", preis="95,000000"),
            _movement(transaktion="Ausbuchung", anzahl="0,5000", preis="90,000000"),
        )
    )

    assert export.kind == "transactions"
    assert export.positions == ()
    assert export.movements[0]["movement"] == "Kauf"
    assert export.movements[0]["movement_kind"] == "buy"
    assert [m["movement_kind"] for m in export.movements] == [
        "buy",
        "sell",
        "dividend",
        "transfer_in",
        "transfer_out",
    ]
    # A dividend row carries a quantity of its own; it must never become a holding.
    assert export.movements[2]["quantity"] == pytest.approx(0.5)
    assert export.movements[2]["taxes"] == pytest.approx(0.45)
    assert all("asset_type" not in movement for movement in export.movements)
    assert all("accrued_interest" in movement for movement in export.movements)
    assert export.movements[0]["date"] == "2026-09-01"
    assert export.movements[0]["accrued_interest"] == pytest.approx(0.0)


def test_a_transactions_export_refuses_to_pose_as_a_portfolio():
    export = parse_export(_transactions_export(_movement(transaktion="Kauf")))

    with pytest.raises(ExtraEtfFormatError, match="movements, not positions"):
        export.require_positions()


def test_unknown_transaction_type_is_carried_through_not_invented():
    export = parse_export(_transactions_export(_movement(transaktion="Steuer")))

    assert export.positions == ()
    assert export.movements[0]["movement"] == "Steuer"
    assert export.movements[0]["movement_kind"] is None


def test_invalid_transaction_date_is_refused():
    with pytest.raises(ExtraEtfFormatError, match="DD.MM.YYYY"):
        parse_export(_transactions_export(_movement(datum="2026-09-01")))

    with pytest.raises(ExtraEtfFormatError, match="not a real date"):
        parse_export(_transactions_export(_movement(datum="31.02.2026")))


def test_crypto_identity_is_a_pair_not_an_isin():
    export = parse_export(
        _holdings_export(
            _holding(
                isin="BTC_to_EUR",
                name="Bitcoin",
                instrument_type="Währung / Krypto",
                region="Global",
                sektor="Diversifizierte Kapitalmärkte",
                portfolio="Example crypto wallet",
                portfolio_id="1000002",
            )
        )
    )

    (position,) = export.require_positions()
    assert position["instrument_id"] == "BTC_to_EUR"
    assert position["instrument_id_kind"] == "crypto_pair"
    assert position["instrument_id_checksum_ok"] is None
    assert position["asset_type"] == "crypto"
    assert position["portfolio_id"] == "1000002"


@pytest.mark.parametrize("identifier", ["BTC", "", "  ", "NOT*AN*ID"])
def test_unattributable_holding_identifier_is_refused(identifier):
    """A bare symbol cannot be resolved, so the import fails instead of guessing."""
    with pytest.raises(ExtraEtfFormatError, match="neither an ISIN nor"):
        parse_export(_holdings_export(_holding(isin=identifier)))


def test_invalid_isin_check_digit_is_reported_not_enforced():
    export = parse_export(
        _holdings_export(_holding(isin=VALID_ISIN), _holding(isin=PLACEHOLDER_ISIN))
    )

    valid, placeholder = export.require_positions()
    assert valid["instrument_id_checksum_ok"] is True
    assert placeholder["instrument_id_checksum_ok"] is False


def test_unknown_instrument_type_leaves_asset_type_unset():
    export = parse_export(_holdings_export(_holding(instrument_type="Anleihe")))

    (position,) = export.require_positions()
    assert position["instrument_type"] == "Anleihe"
    assert position["asset_type"] is None


def test_missing_required_field_is_refused():
    for kwarg, message in (
        ("name", "Name is empty"),
        ("instrument_type", "Typ is empty"),
        ("waehrung", "Währung is empty"),
        ("portfolio_id", "Portfolio ID is empty"),
    ):
        with pytest.raises(ExtraEtfFormatError, match=message):
            parse_export(_holdings_export(_holding(**{kwarg: ""})))


def test_invalid_currency_is_refused():
    with pytest.raises(ExtraEtfFormatError, match="three-letter currency"):
        parse_export(_holdings_export(_holding(waehrung="EURO")))


def test_unknown_column_makes_the_export_unrecognisable():
    row = _holding()
    header = ";".join([*HOLDINGS_COLUMNS, "Neue Spalte"])
    body = ";".join([*(row[column] for column in HOLDINGS_COLUMNS), "x"])

    with pytest.raises(
        ExtraEtfFormatError, match="unexpected columns: \\['Neue Spalte'\\]"
    ):
        parse_export(f"{header}\n{body}\n")


def test_missing_column_makes_the_export_unrecognisable():
    columns = [column for column in HOLDINGS_COLUMNS if column != "Anzahl"]
    row = _holding()

    with pytest.raises(ExtraEtfFormatError, match="missing columns: \\['Anzahl'\\]"):
        parse_export(_export(columns, [row]))


def test_ragged_row_is_refused_rather_than_partially_imported():
    """One malformed row fails the import; it never yields a short portfolio."""
    row = _holding()
    short = ";".join(row[column] for column in HOLDINGS_COLUMNS[:-1])
    long = ";".join([*(row[column] for column in HOLDINGS_COLUMNS), "extra"])

    for ragged in (short, long):
        with pytest.raises(ExtraEtfFormatError, match="cannot be aligned"):
            parse_export(_holdings_export(_holding(), _holding()) + ragged + "\n")


def test_reordered_columns_do_not_mis_align_values():
    """Rows are keyed by column name, so a reordered export stays correct."""
    columns = list(reversed(HOLDINGS_COLUMNS))

    export = parse_export(_export(columns, [_holding()]))

    (position,) = export.require_positions()
    assert position["currency"] == "EUR"
    assert position["quantity"] == pytest.approx(12.3456)
    assert position["name"] == "Example World ETF"
    assert position["portfolio_id"] == "1000001"


def test_duplicate_rows_are_preserved_rather_than_merged():
    export = parse_export(
        _holdings_export(_holding(anzahl="1,0000"), _holding(anzahl="2,0000"))
    )

    assert [position["quantity"] for position in export.require_positions()] == [
        pytest.approx(1.0),
        pytest.approx(2.0),
    ]


def test_empty_holdings_export_is_valid_and_not_an_error():
    """An empty portfolio and an unreadable file are different states."""
    export = parse_export(_holdings_export())

    assert export.kind == "holdings"
    assert export.require_positions() == ()


def test_empty_file_is_refused():
    for empty in ("", "\n\n  \n"):
        with pytest.raises(ExtraEtfFormatError, match="export is empty"):
            parse_export(empty)


def test_as_of_comes_from_the_file_mtime_and_is_labelled(tmp_path):
    path = tmp_path / "holdings.csv"
    path.write_text(_holdings_export(_holding()), encoding="utf-8")
    stamp = 1_760_000_000
    os.utime(path, (stamp, stamp))

    export = read_export(path)

    expected = datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()
    assert export.as_of_source == "file_mtime"
    assert export.as_of == expected
    assert export.source_path == path
    assert export.require_positions()[0]["updated_at"] == expected


def test_bare_text_reports_no_observation_time():
    export = parse_export(_holdings_export(_holding()))

    assert export.as_of is None
    assert export.as_of_source == "unavailable"
    assert export.require_positions()[0]["updated_at"] is None


def test_an_absent_as_of_cannot_claim_an_origin():
    with pytest.raises(ValueError, match="no observation time"):
        parse_export(_holdings_export(_holding()), as_of_source="file_mtime")


def test_bom_and_umlauts_decode_from_utf8_and_cp1252(tmp_path):
    document = _holdings_export(_holding())

    utf8_path = tmp_path / "utf8.csv"
    utf8_path.write_bytes(("\ufeff" + document).encode("utf-8"))
    assert read_export(utf8_path).require_positions()[0]["currency"] == "EUR"

    cp1252_path = tmp_path / "cp1252.csv"
    cp1252_path.write_bytes(document.encode("cp1252"))
    assert read_export(cp1252_path).require_positions()[0]["currency"] == "EUR"


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(ExtraEtfFormatError, match="cannot read export"):
        read_export(tmp_path / "absent.csv")


def test_reader_imports_only_the_standard_library():
    """No network, no broker SDK, no order path can be reached from here."""
    import ast
    from pathlib import Path

    import src.portfolio.extraetf as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module != "__future__"
        ):
            roots.add(node.module.split(".")[0])

    assert roots == {
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "io",
        "os",
        "pathlib",
        "re",
        "typing",
    }
