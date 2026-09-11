from datetime import date

import pytest

from app.domain import Account, Client, Goal, Holding, Liability
from app.market import FX_TO_SGD, UNIVERSE
from app.portfolio import allocation_by, goal_status, summarise, total_unrealised
from app.seed import BOOK, build_book

TODAY = date(2026, 1, 1)


@pytest.fixture
def simple_client():
    """A hand-computable client: one USD equity, one SGD bond, one liability."""
    return Client(
        client_id="T001",
        name="Test Person",
        email="t@example.com",
        age=40,
        risk_profile="balanced",
        segment="affluent",
        monthly_surplus=1_000,
        accounts=(
            Account(
                "ACC-BRK",
                "Brokerage",
                "brokerage",
                cash=5_000,
                holdings=(
                    Holding("VOO", units=10, avg_cost=400.0),  # USD 528.90 now
                    Holding("A35.SI", units=1_000, avg_cost=1.16),  # SGD 1.16 now
                ),
            ),
            Account("ACC-SAV", "Savings", "savings", cash=20_000),
        ),
        goals=(
            Goal("G1", "Near goal", 12_000, date(2027, 1, 1), 0, "high"),
            Goal("G2", "Funded goal", 5_000, date(2030, 1, 1), 5_000, "low"),
        ),
        liabilities=(Liability("Mortgage", "mortgage", 100_000, 0.03, 900),),
    )


def test_net_worth_converts_foreign_currency_and_nets_off_debt(simple_client):
    """A USD position valued at its USD price would understate net worth by 29%."""
    voo = 10 * UNIVERSE["VOO"].price * FX_TO_SGD["USD"]  # 10 * 528.90 * 1.29
    bond = 1_000 * 1.16
    expected_assets = voo + bond + 5_000 + 20_000

    assert simple_client.total_assets == pytest.approx(expected_assets)
    assert simple_client.net_worth == pytest.approx(expected_assets - 100_000)


def test_allocation_excludes_cash_and_sums_to_one(simple_client):
    """Cash in a savings account is not an asset-class allocation decision."""
    allocation = allocation_by(simple_client, "asset_class")

    assert set(allocation) == {"equity", "bond"}
    assert sum(allocation.values()) == pytest.approx(1.0)
    # VOO (~SGD 6,823) dominates the SGD 1,160 bond position.
    assert allocation["equity"] > allocation["bond"]
    assert list(allocation) == ["equity", "bond"]  # sorted by value, descending


def test_unrealised_gain_is_measured_against_converted_cost_basis(simple_client):
    """Cost basis is in the instrument's own currency and must be converted too."""
    gain, gain_pct = total_unrealised(simple_client)

    voo_gain = 10 * (528.90 - 400.0) * FX_TO_SGD["USD"]
    assert gain == pytest.approx(voo_gain)  # the bond was bought at today's price
    assert gain_pct > 0


def test_goal_status_flags_a_goal_the_client_cannot_afford(simple_client):
    """`on_track` must test affordability, not just whether time remains."""
    near, funded = goal_status(simple_client, TODAY)

    # SGD 12,000 over 12 months = 1,000/month against a 1,000 surplus: just affordable.
    assert near["months_remaining"] == 12
    assert near["required_monthly"] == pytest.approx(1_000)
    assert near["on_track"] is True

    # Already fully funded: nothing more required.
    assert funded["progress"] == 1.0
    assert funded["required_monthly"] == 0.0


def test_a_goal_needing_more_than_the_monthly_surplus_is_behind(simple_client):
    """Same goal, half the time: now unaffordable and must be reported as such."""
    stretched = Client(
        **{
            **simple_client.__dict__,
            "goals": (Goal("G1", "Near goal", 12_000, date(2026, 7, 1), 0, "high"),),
        }
    )

    status = goal_status(stretched, TODAY)[0]

    assert status["required_monthly"] == pytest.approx(2_000)
    assert status["on_track"] is False


def test_a_goal_past_its_target_date_reports_the_full_shortfall(simple_client):
    """Dividing by zero months remaining must not raise."""
    overdue = Client(
        **{
            **simple_client.__dict__,
            "goals": (Goal("G1", "Missed", 12_000, date(2025, 1, 1), 4_000, "high"),),
        }
    )

    status = goal_status(overdue, TODAY)[0]

    assert status["months_remaining"] == 0
    assert status["required_monthly"] == pytest.approx(8_000)


def test_briefing_states_computed_totals_rather_than_raw_objects(simple_client):
    """The model must never have to parse a repr or do its own arithmetic."""
    briefing = summarise(simple_client, TODAY)

    assert "SGD 100,000" in briefing  # the mortgage
    assert "Vanguard S&P 500 ETF" in briefing
    assert "Near goal" in briefing
    assert "BEHIND" not in briefing  # both goals are affordable here
    # No leaked object reprs.
    assert "Holding(" not in briefing and "dataclass" not in briefing


def test_the_seeded_book_is_identical_across_builds():
    """Eval sets and cached indexes assume client C002 never changes."""
    rebuilt = build_book()

    assert list(rebuilt) == list(BOOK)
    for cid, client in rebuilt.items():
        assert client.net_worth == pytest.approx(BOOK[cid].net_worth)
        assert [h.avg_cost for h in client.all_holdings()] == [
            h.avg_cost for h in BOOK[cid].all_holdings()
        ]


def test_no_seeded_client_has_negative_net_worth():
    """Counting a mortgage without the property it secures inverts net worth."""
    for cid, client in BOOK.items():
        assert client.net_worth > 0, f"{cid} ({client.name}) is underwater"


def test_mortgaged_clients_own_the_property_the_debt_is_secured_against():
    for cid, client in BOOK.items():
        has_mortgage = any(li.liability_type == "mortgage" for li in client.liabilities)
        if has_mortgage:
            assert client.properties, f"{cid} has a mortgage but no property"
            assert client.property_value > client.total_liabilities * 0.5


def test_property_lifts_net_worth_but_stays_out_of_the_allocation(simple_client):
    """Illiquid real estate is not an asset-class weighting decision."""
    from app.domain import Property

    with_house = Client(
        **{**simple_client.__dict__, "properties": (Property("Flat", "hdb", 400_000),)}
    )

    assert with_house.net_worth == pytest.approx(simple_client.net_worth + 400_000)
    assert with_house.liquid_assets == pytest.approx(simple_client.liquid_assets)
    assert allocation_by(with_house, "asset_class") == allocation_by(
        simple_client, "asset_class"
    )


def test_every_seeded_client_holds_their_risk_profile_model_portfolio():
    """A conservative retiree holding an aggressive book breaks the demo."""
    for client in BOOK.values():
        classes = allocation_by(client, "asset_class")
        if client.risk_profile == "aggressive":
            assert classes.get("bond", 0.0) == 0.0
            assert classes["equity"] > 0.9
        if client.risk_profile == "conservative":
            assert classes.get("bond", 0.0) > 0.5
