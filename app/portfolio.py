"""Portfolio analytics and the grounding context handed to the model.

``summarise`` is the important function here. Dumping the raw object graph into
the prompt would be both expensive and worse: the model would spend attention
re-deriving totals it could get wrong. Computing every figure in Python and
handing over a compact, already-correct briefing keeps arithmetic out of the
model's hands, which is where accuracy comes from.
"""

from collections import defaultdict
from datetime import date

from app.domain import Client, Holding
from app.market import BASE_CURRENCY


def money(amount: float) -> str:
    return f"{BASE_CURRENCY} {amount:,.0f}"


def pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def allocation_by(client: Client, attribute: str) -> dict[str, float]:
    """Invested value split by an instrument attribute, as fractions of 1.0.

    Cash is excluded: this answers "how is my portfolio invested", and folding
    an SRS cash balance into an 'equity vs bond' split would distort it.
    """
    buckets: dict[str, float] = defaultdict(float)
    for holding in client.all_holdings():
        buckets[getattr(holding.instrument, attribute)] += holding.market_value

    total = sum(buckets.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}


def top_holdings(client: Client, limit: int = 5) -> list[Holding]:
    return sorted(client.all_holdings(), key=lambda h: -h.market_value)[:limit]


def total_unrealised(client: Client) -> tuple[float, float]:
    """Absolute and percentage unrealised gain across every holding."""
    holdings = client.all_holdings()
    gain = sum(h.unrealised_gain for h in holdings)
    cost = sum(h.cost_basis for h in holdings)
    return gain, (gain / cost if cost else 0.0)


def goal_status(client: Client, today: date | None = None) -> list[dict]:
    """Per-goal funding position, with the monthly contribution needed.

    ``on_track`` compares the required contribution against the client's actual
    monthly surplus, so it reflects affordability rather than just arithmetic.
    """
    today = today or date.today()
    statuses = []
    for goal in client.goals:
        required = goal.required_monthly(today)
        statuses.append(
            {
                "goal": goal,
                "progress": goal.progress(),
                "months_remaining": goal.months_remaining(today),
                "required_monthly": required,
                "on_track": required <= client.monthly_surplus,
            }
        )
    return statuses


def summarise(client: Client, today: date | None = None) -> str:
    """Compact, pre-computed briefing on the client's position."""
    today = today or date.today()
    gain, gain_pct = total_unrealised(client)

    lines = [
        f"CLIENT: {client.name} ({client.client_id}), age {client.age}, "
        f"{client.segment}, risk profile: {client.risk_profile}",
        f"As of {today.isoformat()}. All figures in {BASE_CURRENCY}.",
        "",
        "NET WORTH",
        f"  Liquid assets:     {money(client.liquid_assets)}",
        f"  Property:          {money(client.property_value)}",
        f"  Total assets:      {money(client.total_assets)}",
        f"  Total liabilities: {money(client.total_liabilities)}",
        f"  Net worth:         {money(client.net_worth)}",
        f"  Monthly surplus:   {money(client.monthly_surplus)}",
        "",
        "ACCOUNTS",
    ]
    for account in client.accounts:
        detail = f"cash {money(account.cash)}"
        if account.holdings:
            detail += f", invested {money(account.invested_value)}"
        lines.append(
            f"  {account.name} ({account.account_type}): "
            f"{money(account.total_value)} - {detail}"
        )

    if client.properties:
        lines += ["", "PROPERTY"]
        for prop in client.properties:
            lines.append(
                f"  {prop.name} ({prop.property_type}): {money(prop.market_value)}"
            )

    lines += ["", "PORTFOLIO", f"  Unrealised P&L: {money(gain)} ({pct(gain_pct)})"]

    lines.append("  By asset class: " + ", ".join(
        f"{name} {pct(weight)}" for name, weight in allocation_by(client, "asset_class").items()
    ))
    lines.append("  By region: " + ", ".join(
        f"{name} {pct(weight)}" for name, weight in allocation_by(client, "region").items()
    ))

    lines.append("  Holdings:")
    for holding in sorted(client.all_holdings(), key=lambda h: -h.market_value):
        inst = holding.instrument
        lines.append(
            f"    {inst.ticker} {inst.name} - {holding.units:,.2f} units, "
            f"value {money(holding.market_value)}, "
            f"unrealised {pct(holding.unrealised_pct)}"
        )

    if client.liabilities:
        lines += ["", "LIABILITIES"]
        for liability in client.liabilities:
            lines.append(
                f"  {liability.name} ({liability.liability_type}): "
                f"{money(liability.outstanding)} outstanding at "
                f"{pct(liability.annual_rate)}, {money(liability.monthly_payment)}/month"
            )

    lines += ["", "GOALS"]
    for status in goal_status(client, today):
        goal = status["goal"]
        verdict = "on track" if status["on_track"] else "BEHIND"
        lines.append(
            f"  {goal.name} [{goal.priority} priority]: "
            f"{money(goal.saved)} of {money(goal.target_amount)} "
            f"({pct(status['progress'])}) by {goal.target_date.isoformat()}, "
            f"{status['months_remaining']} months left, needs "
            f"{money(status['required_monthly'])}/month, {verdict}"
        )

    return "\n".join(lines)
