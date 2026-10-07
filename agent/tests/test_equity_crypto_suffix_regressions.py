import pandas as pd
import pytest
from backtest.correlation import infer_market, _fetch_price_series


@pytest.mark.parametrize("symbol", ["ABNB", "ABNB.US", "SOL", "SOL.US"])
def test_equity_names_are_not_crypto_quote_suffixes(symbol):
    assert infer_market(symbol) == "us_equity"


@pytest.mark.parametrize(
    "symbol",
    ["BTC-USDT", "ETH/USD", "ETHUSDT", "ETHBTC", "XRPUSDT", "LTCETH", "AVAXBNB"],
)
def test_explicit_crypto_pairs_stay_crypto(symbol):
    assert infer_market(symbol) == "crypto"


def test_airbnb_fetch_uses_equity_chain(monkeypatch):
    from backtest.loaders import registry

    seen = []

    class Loader:
        def is_available(self):
            return True

        def fetch(self, codes, **kwargs):
            seen.extend(codes)
            return {codes[0]: pd.DataFrame({"close": [100, 110]})}

    monkeypatch.setattr(registry, "_ensure_registered", lambda: None)
    monkeypatch.setattr(registry, "FALLBACK_CHAINS", {"us_equity": ["equity"]})
    monkeypatch.setattr(registry, "LOADER_REGISTRY", {"equity": Loader})
    assert list(_fetch_price_series(["ABNB"], "2025-01-01", "2025-01-02")) == ["ABNB"]
    assert seen == ["ABNB.US"]
