"""Deterministic seed data for the AuraWealth demo book of business.

Client profiles are hand-authored rather than randomly generated — a book where
a 26-year-old holds a retiree's bond ladder is the kind of detail that makes a
demo fall apart. Randomness is confined to cost-basis jitter, so each holding
shows a plausible mix of gains and losses.

The generator is seeded, so the same book is produced on every process start.
That matters more than it sounds: evaluation sets, cached retrieval indexes and
regression tests all assume yesterday's client 4 is today's client 4.
"""

import random
from datetime import date

from app.domain import Account, Client, Goal, Holding, Liability, Property

SEED = 20260101

# Target asset-class weights per risk profile, mirroring the four models in
# data/kb/internal-model-portfolios.md. This is the table drift is measured
# against, because that is what the policy says drift means: "the absolute
# difference between the actual weight of an asset class and its model weight".
#
# The policy names only equity, bonds and commodities. Listed REITs are counted
# as equity here -- they are listed equity that happens to hold property, they
# carry equity drawdowns, and leaving them in a fourth bucket would make every
# model's weights fail to sum to 1.
MODEL_ALLOCATIONS: dict[str, dict[str, float]] = {
    "conservative": {"equity": 0.30, "bond": 0.60, "commodity": 0.10},
    "balanced": {"equity": 0.60, "bond": 0.35, "commodity": 0.05},
    "growth": {"equity": 0.80, "bond": 0.15, "commodity": 0.05},
    "aggressive": {"equity": 0.95, "bond": 0.00, "commodity": 0.05},
}

# How an instrument's asset class maps onto the three buckets the models use.
MODEL_BUCKET = {"equity": "equity", "reit": "equity", "bond": "bond", "commodity": "commodity"}

# Target weights per risk profile. Weights sum to 1.0 within each profile.
MODEL_PORTFOLIOS: dict[str, dict[str, float]] = {
    "conservative": {
        "A35.SI": 0.30, "SGB2030": 0.20, "AGG": 0.15,
        "D05.SI": 0.10, "C38U.SI": 0.10, "VWRA.L": 0.10, "GLD": 0.05,
    },
    "balanced": {
        "VWRA.L": 0.25, "VOO": 0.15, "D05.SI": 0.12, "O39.SI": 0.08,
        "A17U.SI": 0.10, "A35.SI": 0.15, "AGG": 0.10, "GLD": 0.05,
    },
    "growth": {
        "VOO": 0.28, "QQQ": 0.18, "VWRA.L": 0.16, "AAXJ": 0.10,
        "IEUR": 0.08, "D05.SI": 0.08, "A17U.SI": 0.06, "AGG": 0.06,
    },
    "aggressive": {
        "QQQ": 0.34, "VOO": 0.24, "AAXJ": 0.16, "VWRA.L": 0.12,
        "IEUR": 0.08, "GLD": 0.06,
    },
}

# client_id, name, age, risk profile, segment, investable, cash, monthly surplus
_CLIENT_PROFILES = [
    ("C001", "Priya Raman", "priya.raman@example.com", 34, "growth", "mass affluent", 185_000, 24_000, 3_200),
    ("C002", "Wei Ling Tan", "weiling.tan@example.com", 41, "balanced", "affluent", 620_000, 85_000, 6_800),
    ("C003", "Marcus Lim", "marcus.lim@example.com", 28, "aggressive", "mass affluent", 78_000, 12_500, 2_400),
    ("C004", "Siti Nurhaliza", "siti.n@example.com", 52, "conservative", "affluent", 940_000, 130_000, 7_500),
    ("C005", "Daniel Ong", "daniel.ong@example.com", 45, "balanced", "high net worth", 2_350_000, 310_000, 18_000),
    ("C006", "Aisha Rahman", "aisha.rahman@example.com", 37, "growth", "affluent", 410_000, 46_000, 5_100),
    ("C007", "Kenji Yamamoto", "kenji.y@example.com", 61, "conservative", "high net worth", 3_100_000, 480_000, 9_200),
    ("C008", "Grace Chua", "grace.chua@example.com", 30, "aggressive", "mass affluent", 96_000, 15_000, 2_900),
]

# client_id -> goals
_GOALS: dict[str, list[tuple[str, str, float, date, float, str]]] = {
    "C001": [
        ("G1", "Home down payment", 200_000, date(2029, 6, 1), 61_000, "high"),
        ("G2", "Retirement at 60", 1_400_000, date(2052, 1, 1), 148_000, "medium"),
    ],
    "C002": [
        ("G1", "Children's university fund", 320_000, date(2035, 8, 1), 141_000, "high"),
        ("G2", "Retirement at 60", 2_200_000, date(2045, 1, 1), 505_000, "high"),
        ("G3", "Sabbatical year", 90_000, date(2028, 3, 1), 34_000, "low"),
    ],
    "C003": [
        ("G1", "Emergency fund", 45_000, date(2027, 1, 1), 12_500, "high"),
        ("G2", "First property", 180_000, date(2032, 9, 1), 41_000, "medium"),
    ],
    "C004": [
        ("G1", "Retirement at 62", 2_400_000, date(2036, 5, 1), 1_010_000, "high"),
        ("G2", "Parents' care fund", 250_000, date(2029, 1, 1), 178_000, "high"),
    ],
    "C005": [
        ("G1", "Retirement at 55", 5_000_000, date(2036, 7, 1), 2_540_000, "high"),
        ("G2", "Second property", 900_000, date(2030, 1, 1), 388_000, "medium"),
        ("G3", "Legacy fund", 1_000_000, date(2050, 1, 1), 122_000, "low"),
    ],
    "C006": [
        ("G1", "Career break fund", 120_000, date(2028, 9, 1), 58_000, "medium"),
        ("G2", "Retirement at 58", 1_800_000, date(2047, 4, 1), 402_000, "high"),
    ],
    "C007": [
        ("G1", "Retirement income floor", 3_500_000, date(2027, 1, 1), 3_240_000, "high"),
        ("G2", "Grandchildren's education", 400_000, date(2038, 1, 1), 96_000, "medium"),
    ],
    "C008": [
        ("G1", "Emergency fund", 40_000, date(2027, 6, 1), 15_000, "high"),
        ("G2", "Wedding", 70_000, date(2028, 11, 1), 21_000, "high"),
        ("G3", "Retirement at 60", 1_500_000, date(2056, 1, 1), 84_000, "low"),
    ],
}

# client_id -> liabilities
_LIABILITIES: dict[str, list[tuple[str, str, float, float, float]]] = {
    "C001": [("HDB mortgage", "mortgage", 312_000, 0.026, 1_480)],
    "C002": [("Condo mortgage", "mortgage", 684_000, 0.031, 3_240), ("Car loan", "car_loan", 42_000, 0.048, 890)],
    "C003": [("Education loan", "education_loan", 28_500, 0.042, 520)],
    "C004": [("Condo mortgage", "mortgage", 215_000, 0.028, 2_100)],
    "C005": [("Condo mortgage", "mortgage", 1_180_000, 0.033, 5_600), ("Investment property loan", "mortgage", 620_000, 0.035, 3_100)],
    "C006": [("HDB mortgage", "mortgage", 268_000, 0.026, 1_320)],
    "C007": [],
    "C008": [("Car loan", "car_loan", 31_000, 0.051, 640), ("Credit card", "credit_card", 4_200, 0.258, 350)],
}

# client_id -> properties. Every mortgage above is secured against one of these;
# without them, net worth for a mortgaged client comes out negative.
_PROPERTIES: dict[str, list[tuple[str, str, float]]] = {
    "C001": [("4-room HDB, Punggol", "hdb", 520_000)],
    "C002": [("Condo, Tampines", "condo", 1_150_000)],
    "C003": [],
    "C004": [("Condo, Bukit Timah", "condo", 1_320_000)],
    "C005": [
        ("Condo, River Valley", "condo", 2_100_000),
        ("Investment condo, Novena", "investment", 1_050_000),
    ],
    "C006": [("5-room HDB, Sengkang", "hdb", 610_000)],
    "C007": [("Condo, Holland Village", "condo", 2_400_000)],
    "C008": [],
}


def _build_accounts(rng: random.Random, profile: str, investable: float, cash: float) -> tuple[Account, ...]:
    """Split investable wealth across a brokerage and retirement accounts.

    CPF and SRS balances are held as cash rather than securities: it keeps the
    withdrawal-restricted portion of net worth visibly distinct from the part
    the client can actually touch.
    """
    # Tilt each position off its model weight before building the book. Without
    # this every client's holdings are generated *from* the model table, so
    # actual weight equals target weight to the last decimal and the drift
    # facts are structurally incapable of firing -- a whole branch of the
    # review that could never be reached. A real book drifts because prices
    # moved; the tilt stands in for that. Seeded, so it stays reproducible.
    tilted = {t: w * rng.uniform(0.55, 1.5) for t, w in MODEL_PORTFOLIOS[profile].items()}
    total = sum(tilted.values())
    weights = {t: w / total for t, w in tilted.items()}

    brokerage_value = investable * 0.72
    srs_value = investable * 0.10
    cpf_value = investable * 0.18

    holdings = []
    for ticker, weight in weights.items():
        from app.market import FX_TO_SGD, UNIVERSE

        instrument = UNIVERSE[ticker]
        price_sgd = instrument.price * FX_TO_SGD[instrument.currency]
        units = round(brokerage_value * weight / price_sgd, 4)
        # Entry price between 22% below and 12% above today's — most positions
        # are up, a few are underwater, which is what a real book looks like.
        cost_ratio = rng.uniform(0.78, 1.12)
        holdings.append(
            Holding(ticker=ticker, units=units, avg_cost=round(instrument.price * cost_ratio, 4))
        )

    return (
        Account("ACC-BRK", "Brokerage", "brokerage", cash=cash * 0.4, holdings=tuple(holdings)),
        Account("ACC-SAV", "Savings", "savings", cash=cash * 0.6),
        Account("ACC-SRS", "SRS", "srs", cash=round(srs_value, 2)),
        Account("ACC-CPF", "CPF Ordinary & Special", "cpf", cash=round(cpf_value, 2)),
    )


def build_book(seed: int = SEED) -> dict[str, Client]:
    """Build the full demo book, keyed by client id."""
    rng = random.Random(seed)
    book: dict[str, Client] = {}

    for cid, name, email, age, profile, segment, investable, cash, surplus in _CLIENT_PROFILES:
        goals = tuple(
            Goal(gid, gname, target, target_date, saved, priority)
            for gid, gname, target, target_date, saved, priority in _GOALS[cid]
        )
        liabilities = tuple(
            Liability(lname, ltype, outstanding, rate, payment)
            for lname, ltype, outstanding, rate, payment in _LIABILITIES[cid]
        )
        properties = tuple(
            Property(pname, ptype, value) for pname, ptype, value in _PROPERTIES[cid]
        )
        book[cid] = Client(
            client_id=cid,
            name=name,
            email=email,
            age=age,
            risk_profile=profile,
            segment=segment,
            monthly_surplus=surplus,
            accounts=_build_accounts(rng, profile, investable, cash),
            goals=goals,
            liabilities=liabilities,
            properties=properties,
        )

    return book


# Built once at import; the book is immutable and identical on every run.
BOOK: dict[str, Client] = build_book()
DEFAULT_CLIENT_ID = "C002"


def get_client(client_id: str = DEFAULT_CLIENT_ID) -> Client:
    return BOOK[client_id]
