"""Read-only reader for extraETF portfolio exports.

extraETF exports two semicolon-delimited, German-locale files:

``holdings`` — the positions of one view at export time::

    ISIN;Name;Typ;Anzahl;Kaufpreis;Aktueller Kurs;Aktueller Wert;Währung;Wechselkurs;Region;Land;Sektor;Portfolioname;Portfolio ID

``transactions`` — the movement history of one view::

    Datum;ISIN;Name;Typ;Transaktion;Anzahl;Preis;Gebühren;Steuern;Währung;Wechselkurs;Portfolioname;Portfolio ID;Stückzinsen

Three properties are deliberate and load-bearing.

**A transactions export can never be read as a portfolio.** Its rows are
movements — ``Kauf``/``Verkauf``/``Dividende``/``Einbuchung``/``Ausbuchung`` —
not holdings. An allocation computed from trades looks perfectly plausible and
is wrong, so the two formats are told apart by their header and
:meth:`ExtraEtfExport.require_positions` refuses anything that is not a
holdings export. Movements carry an explicit ``movement_kind`` so a dividend or
a transfer is never mistaken for a trade.

**A holdings export carries no date.** The format has no snapshot column, so
freshness can only come from the file's own modification time, and the export
says so through ``as_of_source``. This reader never claims an observation time
it did not read.

**Validation fails closed.** A row this reader does not recognise raises
:class:`ExtraEtfFormatError` instead of being skipped: an import computed from
90% of a portfolio reads as entirely plausible, whereas a refused import is
merely annoying. Numbers are parsed as German — ``1.386,608`` is 1386.608 — and
a value that cannot be read unambiguously is refused rather than guessed.
``1,234.56`` is US-formatted; reading it as German would be wrong by three
orders of magnitude, so it is refused instead.

Rows are read by column name, so a reordered export cannot mis-align a value
into the wrong column. Rows are **not** merged or de-duplicated: two rows for
the same instrument in one portfolio are returned as two positions, because
summing them silently and dropping one silently are both wrong, and which one
the user meant is not this module's call.

The module writes nothing, holds no credentials, and touches no network. It
only reads a local file the user exported themselves, and it has no path to any
order surface.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

BROKER = "extraetf"

_HOLDINGS_COLUMNS: tuple[str, ...] = (
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
)

_TRANSACTIONS_COLUMNS: tuple[str, ...] = (
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
)

ExportKind = Literal["holdings", "transactions"]

_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_CRYPTO_PAIR_PATTERN = re.compile(r"^[A-Z0-9]{2,12}_to_[A-Z0-9]{2,12}$", re.IGNORECASE)
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_DATE_PATTERN = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")
_NUMERAL_PATTERN = re.compile(r"^[0-9.,]+$")

#: ``Typ`` is an instrument class, not a row shape: a class this reader does not
#: know still describes a real position, so it is carried through with
#: ``asset_type`` left ``None`` rather than refused or guessed at.
_ASSET_TYPE_BY_EXPORT_TYPE = {
    "etf": "etf",
    "aktie": "stock",
    "währung / krypto": "crypto",
}

_MOVEMENT_KIND_BY_TRANSACTION = {
    "kauf": "buy",
    "verkauf": "sell",
    "dividende": "dividend",
    "einbuchung": "transfer_in",
    "ausbuchung": "transfer_out",
}


class ExtraEtfFormatError(ValueError):
    """Raised when an export cannot be read without guessing."""


@dataclass(frozen=True)
class ExtraEtfExport:
    """One parsed export, with its provenance attached.

    Attributes:
        kind: Which of the two extraETF formats this file is.
        positions: Holdings rows. Always empty for a transactions export.
        movements: Transactions rows. Always empty for a holdings export.
        as_of: ISO-8601 observation time, or ``None`` when it could not be read.
        as_of_source: Where ``as_of`` came from — ``"file_mtime"`` for a file
            read from disk, ``"unavailable"`` for bare text with no timestamp.
            A holdings export has no date column, so ``file_mtime`` is an
            inference from the file's own modification time and is labelled as
            one rather than presented as a recorded snapshot date.
        source_path: The file this export was read from, when there was one.
    """

    kind: ExportKind
    positions: tuple[dict[str, Any], ...]
    movements: tuple[dict[str, Any], ...]
    as_of: str | None
    as_of_source: Literal["file_mtime", "unavailable"]
    source_path: Path | None

    def require_positions(self) -> tuple[dict[str, Any], ...]:
        """Return holdings positions, refusing any other export outright.

        Returns:
            The parsed positions, possibly empty for an empty portfolio.

        Raises:
            ExtraEtfFormatError: If this export is a transactions export, whose
                rows are movements rather than holdings.
        """
        if self.kind != "holdings":
            raise ExtraEtfFormatError(
                "this is a transactions export and holds movements, not positions; "
                "importing it as a portfolio would build an allocation out of trades"
            )
        return self.positions


def read_export(path: str | Path) -> ExtraEtfExport:
    """Read and parse one extraETF export file.

    The file's modification time becomes ``as_of`` with
    ``as_of_source="file_mtime"``, because neither export format carries a
    snapshot-observation column.

    Args:
        path: The exported ``.csv`` file.

    Returns:
        The parsed export.

    Raises:
        ExtraEtfFormatError: If the file cannot be read, decoded, or parsed.
    """
    resolved = Path(path)
    try:
        # Stat the same descriptor the bytes came from: a separate ``stat`` call
        # can observe a later write than the read did, which would let the import
        # claim a fresher observation time than the content it actually holds.
        with resolved.open("rb") as handle:
            data = handle.read()
            modified = os.fstat(handle.fileno()).st_mtime
    except OSError as exc:
        raise ExtraEtfFormatError(f"cannot read export {resolved}: {exc}") from exc
    as_of = datetime.fromtimestamp(modified, tz=timezone.utc).isoformat()
    return parse_export(
        data, source_path=resolved, as_of=as_of, as_of_source="file_mtime"
    )


def parse_export(
    data: str | bytes,
    *,
    source_path: str | Path | None = None,
    as_of: str | None = None,
    as_of_source: Literal["file_mtime", "unavailable"] = "unavailable",
) -> ExtraEtfExport:
    """Parse one extraETF export from text or raw bytes.

    Args:
        data: The export contents, as text or as raw file bytes.
        source_path: The originating file, recorded for provenance only.
        as_of: ISO-8601 observation time, when the caller knows one.
        as_of_source: Where ``as_of`` came from.

    Returns:
        The parsed export. An export with a header and no data rows is valid and
        yields no positions — an empty portfolio is not an unreadable one.

    Raises:
        ExtraEtfFormatError: If the header is not one of the two known formats,
            or any row cannot be read without guessing.
        ValueError: If ``as_of_source`` claims an origin for an absent ``as_of``.
    """
    if as_of is None and as_of_source != "unavailable":
        raise ValueError(
            "as_of_source cannot claim where an observation time came from when "
            "there is no observation time"
        )
    text = _decode(data)
    kind, header, rows = _read_rows(text)
    if kind == "holdings":
        positions = tuple(
            _parse_holding(header, cells, number, as_of=as_of) for number, cells in rows
        )
        movements: tuple[dict[str, Any], ...] = ()
    else:
        positions = ()
        movements = tuple(
            _parse_movement(header, cells, number) for number, cells in rows
        )
    return ExtraEtfExport(
        kind=kind,
        positions=positions,
        movements=movements,
        as_of=as_of,
        as_of_source=as_of_source,
        source_path=Path(source_path) if source_path is not None else None,
    )


def _decode(data: str | bytes) -> str:
    """Decode export bytes, preferring UTF-8 and falling back to cp1252.

    Args:
        data: Export contents as text or bytes.

    Returns:
        The decoded text.

    Raises:
        ExtraEtfFormatError: If neither encoding decodes the bytes.
    """
    if isinstance(data, str):
        return data
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtraEtfFormatError("export is decodable as neither UTF-8 nor cp1252")


def _read_rows(
    text: str,
) -> tuple[ExportKind, tuple[str, ...], list[tuple[int, list[str]]]]:
    """Split an export into its format, header and data rows.

    Args:
        text: The decoded export.

    Returns:
        The detected format, the header columns, and ``(line number, cells)``
        pairs for every non-blank data row.

    Raises:
        ExtraEtfFormatError: If the export is empty, its header is not one of the
            two known formats, or a row cannot be aligned to that header.
    """
    reader = csv.reader(io.StringIO(text), delimiter=";")
    header: tuple[str, ...] | None = None
    number = 0
    for cells in reader:
        number += 1
        if not any(cell.strip() for cell in cells):
            continue
        header = tuple(cell.strip().lstrip("\ufeff") for cell in cells)
        break
    if header is None:
        raise ExtraEtfFormatError("export is empty")
    kind = _detect_kind(header)

    rows: list[tuple[int, list[str]]] = []
    for cells in reader:
        number += 1
        if not any(cell.strip() for cell in cells):
            continue
        if len(cells) != len(header):
            raise ExtraEtfFormatError(
                f"row {number}: expected {len(header)} fields to match the header, "
                f"found {len(cells)}; a row that cannot be aligned is refused rather "
                f"than skipped"
            )
        rows.append((number, [cell.strip() for cell in cells]))
    return kind, header, rows


def _detect_kind(header: tuple[str, ...]) -> ExportKind:
    """Identify which extraETF format a header belongs to.

    Args:
        header: The export's column names.

    Returns:
        ``"holdings"`` or ``"transactions"``.

    Raises:
        ExtraEtfFormatError: If the header is neither known format exactly.
    """
    columns = set(header)
    if len(header) == len(_HOLDINGS_COLUMNS) and columns == set(_HOLDINGS_COLUMNS):
        return "holdings"
    if len(header) == len(_TRANSACTIONS_COLUMNS) and columns == set(
        _TRANSACTIONS_COLUMNS
    ):
        return "transactions"
    transactions_like = "Transaktion" in columns
    expected = _TRANSACTIONS_COLUMNS if transactions_like else _HOLDINGS_COLUMNS
    missing = [column for column in expected if column not in columns]
    unexpected = [column for column in header if column not in expected]
    raise ExtraEtfFormatError(
        "export header is not a recognised extraETF format; against the "
        f"{'transactions' if transactions_like else 'holdings'} format, missing "
        f"columns: {missing or 'none'}, unexpected columns: {unexpected or 'none'}"
    )


def _parse_holding(
    header: tuple[str, ...], cells: list[str], number: int, *, as_of: str | None
) -> dict[str, Any]:
    """Convert one holdings row into a position.

    Args:
        header: The export's column names, which key the row's fields.
        cells: The row's fields.
        number: The row's line number, for error messages.
        as_of: The export's observation time, mirrored onto the position.

    Returns:
        One position in the portfolio wire shape.

    Raises:
        ExtraEtfFormatError: If the row cannot be read without guessing.
    """
    row = dict(zip(header, cells, strict=True))
    identifier, identifier_kind, checksum_ok = _describe_identifier(row["ISIN"])
    if identifier_kind not in {"isin", "crypto_pair"}:
        raise ExtraEtfFormatError(
            f"row {number}: {row['ISIN']!r} is neither an ISIN nor an extraETF crypto "
            f"pair such as BTC_to_EUR, so the holding cannot be attributed to an "
            f"instrument"
        )
    name = _required(row, "Name", number)
    instrument_type = _required(row, "Typ", number)
    currency = _required(row, "Währung", number).upper()
    if not _CURRENCY_PATTERN.fullmatch(currency):
        raise ExtraEtfFormatError(
            f"row {number}: Währung is not a three-letter currency code: "
            f"{row['Währung']!r}"
        )
    portfolio_id = _required(row, "Portfolio ID", number)
    quantity = _parse_decimal(row["Anzahl"], field="Anzahl", row_number=number)
    if quantity is None:
        raise ExtraEtfFormatError(f"row {number}: Anzahl is empty")
    return {
        "broker": BROKER,
        "source": "extraetf_export",
        "symbol": identifier,
        "quote_symbol": identifier,
        "instrument_id": identifier,
        "instrument_id_kind": identifier_kind,
        "instrument_id_checksum_ok": checksum_ok,
        "name": name,
        "instrument_type": instrument_type,
        "asset_type": _ASSET_TYPE_BY_EXPORT_TYPE.get(instrument_type.strip().lower()),
        "market": None,
        "currency": currency,
        "price_currency": currency,
        "quantity": _number(quantity),
        "cost_price": _optional_number(
            row["Kaufpreis"], field="Kaufpreis", row_number=number
        ),
        "market_price": _optional_number(
            row["Aktueller Kurs"], field="Aktueller Kurs", row_number=number
        ),
        "source_market_value": _optional_number(
            row["Aktueller Wert"], field="Aktueller Wert", row_number=number
        ),
        "fx_rate": _optional_number(
            row["Wechselkurs"], field="Wechselkurs", row_number=number
        ),
        "region": row["Region"] or None,
        "country": row["Land"] or None,
        "sector": row["Sektor"] or None,
        "portfolio_id": portfolio_id,
        "portfolio_name": row["Portfolioname"] or None,
        "updated_at": as_of,
    }


def _parse_movement(
    header: tuple[str, ...], cells: list[str], number: int
) -> dict[str, Any]:
    """Convert one transactions row into a movement.

    Args:
        header: The export's column names, which key the row's fields.
        cells: The row's fields.
        number: The row's line number, for error messages.

    Returns:
        One movement, carrying the export's own transaction label plus a
        normalised ``movement_kind``.

    Raises:
        ExtraEtfFormatError: If the row cannot be read without guessing.
    """
    row = dict(zip(header, cells, strict=True))
    movement = _required(row, "Transaktion", number)
    transaction_date = _required(row, "Datum", number)
    match = _DATE_PATTERN.fullmatch(transaction_date)
    if match is None:
        raise ExtraEtfFormatError(
            f"row {number}: Datum is not a DD.MM.YYYY date: {transaction_date!r}"
        )
    day, month, year = (int(part) for part in match.groups())
    try:
        booked_on = datetime(year, month, day).date().isoformat()
    except ValueError as exc:
        raise ExtraEtfFormatError(
            f"row {number}: Datum is not a real date: {transaction_date!r}"
        ) from exc
    currency = _required(row, "Währung", number).upper()
    if not _CURRENCY_PATTERN.fullmatch(currency):
        raise ExtraEtfFormatError(
            f"row {number}: Währung is not a three-letter currency code: "
            f"{row['Währung']!r}"
        )
    identifier, identifier_kind, checksum_ok = _describe_identifier(row["ISIN"])
    quantity = _parse_decimal(row["Anzahl"], field="Anzahl", row_number=number)
    if quantity is None:
        raise ExtraEtfFormatError(f"row {number}: Anzahl is empty")
    return {
        "broker": BROKER,
        "source": "extraetf_export",
        "date": booked_on,
        "instrument_id": identifier or None,
        "instrument_id_kind": identifier_kind,
        "instrument_id_checksum_ok": checksum_ok,
        "name": _required(row, "Name", number),
        "instrument_type": _required(row, "Typ", number),
        "movement": movement,
        "movement_kind": _MOVEMENT_KIND_BY_TRANSACTION.get(movement.strip().lower()),
        "quantity": _number(quantity),
        "price": _optional_number(row["Preis"], field="Preis", row_number=number),
        "fees": _optional_number(row["Gebühren"], field="Gebühren", row_number=number),
        "taxes": _optional_number(row["Steuern"], field="Steuern", row_number=number),
        "currency": currency,
        "fx_rate": _optional_number(
            row["Wechselkurs"], field="Wechselkurs", row_number=number
        ),
        "accrued_interest": _optional_number(
            row["Stückzinsen"], field="Stückzinsen", row_number=number
        ),
        "portfolio_id": _required(row, "Portfolio ID", number),
        "portfolio_name": row["Portfolioname"] or None,
    }


def _required(row: dict[str, str], column: str, number: int) -> str:
    """Return a column's value, refusing an empty one.

    Args:
        row: The aligned row.
        column: The column to read.
        number: The row's line number, for error messages.

    Returns:
        The non-empty value.

    Raises:
        ExtraEtfFormatError: If the value is empty.
    """
    value = row[column]
    if not value:
        raise ExtraEtfFormatError(f"row {number}: {column} is empty")
    return value


def _optional_number(raw: str, *, field: str, row_number: int) -> float | None:
    """Parse an optional German-formatted number.

    Args:
        raw: The raw field text.
        field: The column name, for error messages.
        row_number: The row's line number, for error messages.

    Returns:
        The value, or ``None`` when the field is blank. A blank price is an
        absent observation rather than a zero, and is never guessed at.

    Raises:
        ExtraEtfFormatError: If the field is present but unreadable.
    """
    value = _parse_decimal(raw, field=field, row_number=row_number)
    return None if value is None else _number(value)


def _parse_decimal(raw: str, *, field: str, row_number: int) -> Decimal | None:
    """Parse a German-formatted number, refusing anything ambiguous.

    A comma is the decimal separator and a period may only be a thousands
    separator: ``1.386,608`` is 1386.608 and ``12,3456`` is 12.3456. A value
    whose separators are US-ordered, or whose period grouping is not a plain
    thousands separator, is refused rather than guessed at, because guessing
    wrong changes a position's size by orders of magnitude.

    Args:
        raw: The raw field text.
        field: The column name, for error messages.
        row_number: The row's line number, for error messages.

    Returns:
        The parsed value, or ``None`` when the field is blank.

    Raises:
        ExtraEtfFormatError: If the field is unreadable or ambiguous.
    """
    text = raw.strip()
    if not text:
        return None
    sign = ""
    if text[0] in "+-":
        sign, text = text[0], text[1:]
    if not text or not _NUMERAL_PATTERN.fullmatch(text):
        raise ExtraEtfFormatError(f"row {row_number}: {field} is not a number: {raw!r}")
    whole, comma, fraction = text.rpartition(",")
    if not comma:
        whole, fraction = text, ""
    elif "." in fraction:
        raise ExtraEtfFormatError(
            f"row {row_number}: {field} reads as a US-formatted number: {raw!r}; "
            f"extraETF exports use German formatting such as 1.386,608"
        )
    groups = whole.split(".")
    if (
        any(not group.isdigit() for group in groups)
        or any(len(group) != 3 for group in groups[1:])
        or (comma and not fraction.isdigit())
    ):
        raise ExtraEtfFormatError(
            f"row {row_number}: {field} has ambiguous separators: {raw!r}; expected "
            f"German formatting such as 1.386,608"
        )
    digits = "".join(groups) + (f".{fraction}" if comma else "")
    try:
        return Decimal(f"{sign}{digits}")
    except InvalidOperation as exc:  # pragma: no cover - guarded by the checks above
        raise ExtraEtfFormatError(
            f"row {row_number}: {field} is not a number: {raw!r}"
        ) from exc


def _describe_identifier(raw: str) -> tuple[str, str, bool | None]:
    """Describe an export identifier without refusing anything.

    Args:
        raw: The raw ``ISIN`` column value.

    Returns:
        The identifier as the export spelled it, its kind — ``"isin"``,
        ``"crypto_pair"``, ``"raw_symbol"`` or ``"unknown"`` — and, for an
        ISIN-shaped value, whether its check digit is valid. extraETF stores
        crypto identifiers that are not ISINs, so the check digit is reported
        rather than enforced: refusing a whole import over one unverifiable
        identifier would block a user whose positions are otherwise perfectly
        readable.

        An ISIN is returned upper-cased, because an ISIN is canonical and
        case-insensitive. A crypto pair such as ``BTC_to_EUR`` is returned
        exactly as exported, because it is extraETF's own label and not a
        canonical code — rewriting its case would invent an identity we do not
        own.
    """
    identifier = raw.strip()
    if not identifier:
        return "", "unknown", None
    canonical = identifier.upper()
    if _ISIN_PATTERN.fullmatch(canonical):
        return canonical, "isin", _isin_checksum_ok(canonical)
    if _CRYPTO_PAIR_PATTERN.fullmatch(identifier):
        return identifier, "crypto_pair", None
    return identifier, "raw_symbol", None


def _isin_checksum_ok(value: str) -> bool:
    """Check an ISIN's check digit.

    Args:
        value: A value already matching :data:`_ISIN_PATTERN`.

    Returns:
        ``True`` when the Luhn-style check digit is valid.
    """
    digits = "".join(
        str(ord(character) - 55) if character.isalpha() else character
        for character in value
    )
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = int(character)
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _number(value: Decimal) -> float:
    """Return a wire-ready float, quantized as the connector path does.

    Args:
        value: The parsed value.

    Returns:
        The value as a float, quantized to eight decimal places to match
        :func:`src.portfolio.normalization.normalize_position`.
    """
    return float(value.quantize(Decimal("0.00000001")))
