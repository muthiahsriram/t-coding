"""Core domain model for a client's financial position.

Money is held as ``float``. These are market valuations, not ledger balances —
already approximations that move by the second — and every figure is rounded at
presentation. Anything that had to settle to the cent would want ``Decimal``.
"""

from dataclasses import dataclass, field
from datetime import date

from app.market import FX_TO_SGD, UNIVERSE


@dataclass(frozen=True)
class Holding:
    ticker: str
    units: float
    avg_cost: float  # per unit, in the instrument's own currency

    @property
    def instrument(self):
        return UNIVERSE[self.ticker]

    @property
    def market_value(self) -> float:
        """Current value in the base currency."""
        inst = self.instrument
        return self.units * inst.price * FX_TO_SGD[inst.currency]

    @property
    def cost_basis(self) -> float:
        inst = self.instrument
        return self.units * self.avg_cost * FX_TO_SGD[inst.currency]

    @property
    def unrealised_gain(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealised_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealised_gain / self.cost_basis


@dataclass(frozen=True)
class Account:
    account_id: str
    name: str
    # savings | brokerage | cpf | srs — CPF and SRS are Singapore retirement
    # schemes and carry withdrawal restrictions the advisor must respect.
    account_type: str
    cash: float = 0.0
    holdings: tuple[Holding, ...] = ()

    @property
    def invested_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def total_value(self) -> float:
        return self.cash + self.invested_value


@dataclass(frozen=True)
class Property:
    """Real estate held directly.

    Carried separately from investment accounts: property is illiquid and is
    not part of any allocation decision, but omitting it while still counting
    the mortgage against it makes net worth wildly negative.
    """

    name: str
    property_type: str  # hdb | condo | investment
    market_value: float


@dataclass(frozen=True)
class Liability:
    name: str
    liability_type: str  # mortgage | car_loan | credit_card | education_loan
    outstanding: float
    annual_rate: float
    monthly_payment: float


@dataclass(frozen=True)
class Goal:
    goal_id: str
    name: str
    target_amount: float
    target_date: date
    saved: float
    priority: str  # high | medium | low

    def progress(self) -> float:
        """Fraction of the target already funded, capped at 1.0."""
        if self.target_amount <= 0:
            return 1.0
        return min(self.saved / self.target_amount, 1.0)

    def months_remaining(self, today: date) -> int:
        months = (self.target_date.year - today.year) * 12 + (
            self.target_date.month - today.month
        )
        return max(months, 0)

    def required_monthly(self, today: date) -> float:
        """Contribution needed each month to hit the target on time.

        Ignores investment growth deliberately — a savings rate the client can
        act on today is more useful than one contingent on an assumed return.
        """
        shortfall = max(self.target_amount - self.saved, 0.0)
        months = self.months_remaining(today)
        if shortfall == 0:
            return 0.0
        if months == 0:
            return shortfall
        return shortfall / months


@dataclass(frozen=True)
class Client:
    client_id: str
    name: str
    email: str
    age: int
    risk_profile: str  # conservative | balanced | growth | aggressive
    segment: str  # mass affluent | affluent | high net worth
    monthly_surplus: float  # income less expenses
    accounts: tuple[Account, ...] = ()
    goals: tuple[Goal, ...] = ()
    liabilities: tuple[Liability, ...] = field(default=())
    properties: tuple[Property, ...] = field(default=())

    @property
    def liquid_assets(self) -> float:
        """Assets held in accounts — excludes property."""
        return sum(a.total_value for a in self.accounts)

    @property
    def property_value(self) -> float:
        return sum(p.market_value for p in self.properties)

    @property
    def total_assets(self) -> float:
        return self.liquid_assets + self.property_value

    @property
    def total_liabilities(self) -> float:
        return sum(liability.outstanding for liability in self.liabilities)

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_liabilities

    def all_holdings(self) -> list[Holding]:
        return [h for account in self.accounts for h in account.holdings]
