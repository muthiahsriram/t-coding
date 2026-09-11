"""Deterministic portfolio facts, computed in Python before any model runs.

The analyst agent's job is to decide which of these matter and say so in the
client's terms. It is emphatically *not* to work out drift from a list of
holdings. Every number in the review therefore originates here, where it is
testable, and the model only ever selects and phrases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain import Client
from app.portfolio import allocation_by, goal_status, money, pct
from app.seed import MODEL_ALLOCATIONS, MODEL_BUCKET

# Mirrors data/kb/internal-concentration-limits.md and
# data/kb/internal-model-portfolios.md. Duplicated as code because the review
# must compute against them, not read about them; the tests assert the two
# stay in step.
SINGLE_NAME_LIMIT = 0.15
SINGLE_NAME_WATCH = 0.10
SECTOR_LIMIT = 0.30
HOME_BIAS_LIMIT = 0.50
CASH_DRAG_LIMIT = 0.15
DRIFT_REPORT = 0.05
DRIFT_ACTION = 0.10
EMERGENCY_MONTHS = 6
SHORT_HORIZON_YEARS = 3


@dataclass(frozen=True)
class Fact:
    """One measured observation, with the threshold it was measured against."""

    key: str
    topic: str
    statement: str
    breach: bool

    def render(self) -> str:
        return f"[{self.key}] {'BREACH' if self.breach else 'ok'} - {self.statement}"


def _model_weights(client: Client) -> dict[str, float]:
    return MODEL_ALLOCATIONS.get(client.risk_profile, {})


def drift(client: Client) -> list[Fact]:
    """Asset-class deviation from the client's model portfolio.

    Measured by asset class, not by instrument, because that is what the
    rebalancing policy defines drift to be: "the absolute difference between
    the actual weight of an asset class and its model weight". Measuring it
    per-instrument would report a client who swapped one global equity tracker
    for another as having drifted twice, when their allocation never moved.
    """
    holdings = client.all_holdings()
    invested = sum(h.market_value for h in holdings)
    if invested <= 0:
        return []

    actual: dict[str, float] = {}
    for holding in holdings:
        bucket = MODEL_BUCKET.get(holding.instrument.asset_class, holding.instrument.asset_class)
        actual[bucket] = actual.get(bucket, 0.0) + holding.market_value / invested

    target = _model_weights(client)
    facts = []
    for bucket in sorted(set(actual) | set(target)):
        gap = actual.get(bucket, 0.0) - target.get(bucket, 0.0)
        # Rounded to the precision the statement prints, for the same reason
        # the concentration limit is: a gap that displays as "+5.0 points"
        # must not be silently below the 5-point reporting threshold.
        points = round(gap * 100, 1)
        if abs(points) < DRIFT_REPORT * 100:
            continue
        breach = abs(points) >= DRIFT_ACTION * 100
        facts.append(
            Fact(
                key=f"drift-{bucket}",
                topic="allocation",
                statement=(
                    f"{bucket} is at {pct(actual.get(bucket, 0.0))} against a "
                    f"{client.risk_profile} model weight of {pct(target.get(bucket, 0.0))}, "
                    f"a drift of {points:+.1f} points "
                    f"({'beyond the 10-point action threshold' if breach else 'within the 10-point action threshold'})"
                ),
                breach=breach,
            )
        )
    return facts


def concentration(client: Client) -> list[Fact]:
    holdings = client.all_holdings()
    invested = sum(h.market_value for h in holdings)
    if invested <= 0:
        return []

    by_ticker: dict[str, float] = {}
    for holding in holdings:
        by_ticker[holding.instrument.ticker] = (
            by_ticker.get(holding.instrument.ticker, 0.0) + holding.market_value
        )

    # The limit is on equity positions specifically. Applied to everything, it
    # reports the conservative model's own 60% bond allocation as a permanent
    # breach of the firm's policy, which is not what the policy says.
    equities = {
        h.instrument.ticker
        for h in holdings
        if MODEL_BUCKET.get(h.instrument.asset_class) == "equity"
    }

    facts = []
    for ticker, value in sorted(by_ticker.items(), key=lambda kv: -kv[1]):
        weight = value / invested
        if weight < SINGLE_NAME_WATCH or ticker not in equities:
            continue
        facts.append(
            Fact(
                key=f"concentration-{ticker}",
                topic="concentration",
                statement=(
                    f"{ticker} is {pct(weight)} of invested assets ({money(value)}), "
                    f"against a 15% single-name limit and a 10% reporting threshold"
                ),
                # Rounded to the precision we display: reporting "15.0% of
                # invested assets" as a breach on one line and as compliant on
                # the next, because one is 0.15000001, is indefensible to a
                # client even when it is arithmetically true.
                breach=round(weight, 4) > SINGLE_NAME_LIMIT,
            )
        )

    for region, weight in allocation_by(client, "region").items():
        limit = HOME_BIAS_LIMIT if region == "Singapore" else 0.40
        if weight > limit:
            facts.append(
                Fact(
                    key=f"region-{region.lower().replace(' ', '-')}",
                    topic="concentration",
                    statement=(
                        f"{region} exposure is {pct(weight)} against a "
                        f"{pct(limit)} limit for that market"
                    ),
                    breach=True,
                )
            )

    for asset_class, weight in allocation_by(client, "asset_class").items():
        if asset_class != "equity" and weight > SECTOR_LIMIT:
            facts.append(
                Fact(
                    key=f"sector-{asset_class}",
                    topic="concentration",
                    statement=(
                        f"{asset_class} exposure is {pct(weight)} against a 30% sector limit"
                    ),
                    breach=True,
                )
            )
    return facts


def liquidity(client: Client, today: date) -> list[Fact]:
    cash = sum(account.cash for account in client.accounts)
    investable = cash + sum(h.market_value for h in client.all_holdings())
    facts = []

    if investable > 0:
        weight = cash / investable
        if weight > CASH_DRAG_LIMIT:
            facts.append(
                Fact(
                    key="cash-drag",
                    topic="cash",
                    statement=(
                        f"cash is {money(cash)}, {pct(weight)} of investable assets, "
                        f"above the 15% threshold at which a deployment proposal is required"
                    ),
                    breach=True,
                )
            )

    # The seeded book gives us surplus rather than expenses; surplus is a lower
    # bound on monthly outgoings only in the loosest sense, so this is stated
    # as a ratio to surplus and left for the analyst to qualify.
    months = cash / client.monthly_surplus if client.monthly_surplus else 0.0
    facts.append(
        Fact(
            key="emergency-reserve",
            topic="liquidity",
            statement=(
                f"cash of {money(cash)} covers {months:.1f} months of the client's "
                f"{money(client.monthly_surplus)} monthly surplus; policy requires six "
                f"months of expenses, which are not on file"
            ),
            breach=months < EMERGENCY_MONTHS,
        )
    )
    return facts


def currency(client: Client, today: date) -> list[Fact]:
    holdings = client.all_holdings()
    invested = sum(h.market_value for h in holdings)
    if invested <= 0:
        return []

    foreign = sum(
        h.market_value for h in holdings if h.instrument.currency != "SGD"
    )
    weight = foreign / invested

    near_goals = [
        goal
        for goal in client.goals
        if (goal.target_date - today).days / 365.25 <= SHORT_HORIZON_YEARS
        and goal.saved < goal.target_amount
    ]

    return [
        Fact(
            key="currency-exposure",
            topic="currency",
            statement=(
                f"{pct(weight)} of invested assets ({money(foreign)}) is denominated "
                f"outside SGD, while {len(near_goals)} unfunded goal(s) fall inside "
                f"three years and must be funded in SGD"
            ),
            breach=bool(near_goals) and weight > 0.5,
        )
    ]


def goals(client: Client, today: date) -> list[Fact]:
    facts = []
    for status in goal_status(client, today):
        goal = status["goal"]
        facts.append(
            Fact(
                key=f"goal-{goal.goal_id.lower()}",
                topic="goal",
                statement=(
                    f"{goal.name} [{goal.priority}]: {money(goal.saved)} of "
                    f"{money(goal.target_amount)} ({pct(status['progress'])}), "
                    f"{status['months_remaining']} months left, needs "
                    f"{money(status['required_monthly'])}/month against a "
                    f"{money(client.monthly_surplus)} surplus"
                ),
                breach=not status["on_track"],
            )
        )
    return facts


def portfolio_facts(client: Client, today: date | None = None) -> list[Fact]:
    """Every measured fact about the client, breaches first."""
    today = today or date.today()
    facts = [
        *concentration(client),
        *drift(client),
        *liquidity(client, today),
        *currency(client, today),
        *goals(client, today),
    ]
    return sorted(facts, key=lambda f: (not f.breach, f.key))


def render_facts(facts: list[Fact]) -> str:
    return "\n".join(fact.render() for fact in facts)
