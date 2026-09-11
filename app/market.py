"""Instrument universe and FX rates.

A small curated universe beats randomly generated tickers: allocations stay
plausible, asset-class and region breakdowns mean something, and the numbers
survive a sceptical reader. Prices are a fixed snapshot; the live price stream
overlays them later.
"""

from dataclasses import dataclass

BASE_CURRENCY = "SGD"

# Snapshot rates into the base currency.
FX_TO_SGD = {
    "SGD": 1.0,
    "USD": 1.29,
    "EUR": 1.41,
    "HKD": 0.165,
}


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    asset_class: str  # equity | bond | reit | commodity | cash
    region: str  # Singapore | US | Europe | Asia ex-Japan | Global
    currency: str
    price: float
    annual_yield: float = 0.0  # distribution yield, as a fraction


UNIVERSE: dict[str, Instrument] = {
    i.ticker: i
    for i in [
        # Singapore equities
        Instrument("D05.SI", "DBS Group Holdings", "equity", "Singapore", "SGD", 38.42, 0.058),
        Instrument("O39.SI", "OCBC Bank", "equity", "Singapore", "SGD", 16.28, 0.061),
        Instrument("Z74.SI", "Singtel", "equity", "Singapore", "SGD", 3.19, 0.047),
        # Singapore REITs
        Instrument("A17U.SI", "CapitaLand Ascendas REIT", "reit", "Singapore", "SGD", 2.74, 0.056),
        Instrument("C38U.SI", "CapitaLand Integrated Commercial Trust", "reit", "Singapore", "SGD", 2.11, 0.052),
        # US / global equity funds
        Instrument("VOO", "Vanguard S&P 500 ETF", "equity", "US", "USD", 528.90, 0.013),
        Instrument("QQQ", "Invesco Nasdaq 100 ETF", "equity", "US", "USD", 486.15, 0.006),
        Instrument("VWRA.L", "Vanguard FTSE All-World ETF", "equity", "Global", "USD", 138.60, 0.017),
        Instrument("IEUR", "iShares Core MSCI Europe ETF", "equity", "Europe", "USD", 58.34, 0.028),
        Instrument("AAXJ", "iShares MSCI Asia ex-Japan ETF", "equity", "Asia ex-Japan", "USD", 74.22, 0.021),
        # Fixed income
        Instrument("A35.SI", "ABF Singapore Bond Index Fund", "bond", "Singapore", "SGD", 1.16, 0.031),
        Instrument("AGG", "iShares Core US Aggregate Bond ETF", "bond", "US", "USD", 99.47, 0.038),
        Instrument("SGB2030", "Singapore Savings Bond 2030", "bond", "Singapore", "SGD", 100.00, 0.029),
        # Commodities
        Instrument("GLD", "SPDR Gold Shares", "commodity", "Global", "USD", 241.80, 0.0),
    ]
}


def price_in_sgd(ticker: str) -> float:
    """Current price of ``ticker`` converted to the base currency."""
    instrument = UNIVERSE[ticker]
    return instrument.price * FX_TO_SGD[instrument.currency]
