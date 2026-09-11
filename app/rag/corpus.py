"""Assembles the knowledge base the retrieval layer indexes.

Two sources. The regulation and policy documents under ``data/kb`` are written
by hand: they carry the numbers a wrong answer would be expensive about, and
they are the documents the reranker's recency and authority signals exist to
order. Everything else - product factsheets, market commentary, meeting notes,
FAQs - is generated deterministically from the seeded book.

Generated rather than written because the retrieval eval needs a *fixed*
corpus for NDCG figures to be comparable across runs, and generated rather
than LLM-authored because putting a model call inside the index build makes
the corpus non-reproducible and unbuildable off a live workspace.
"""

from __future__ import annotations

import random
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from app.domain import Client
from app.market import UNIVERSE, Instrument
from app.rag.documents import Chunk, DocMeta, Document, chunk_document
from app.seed import BOOK

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"

CORPUS_SEED = 20260101
TODAY = date(2026, 1, 1)


# --- hand-written documents ------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_META_FIELDS = {f for f in DocMeta.__dataclass_fields__}


def _parse_front_matter(raw: str, fallback_id: str) -> tuple[DocMeta, str]:
    """Split ``key: value`` front matter off a markdown document.

    Hand-rolled rather than PyYAML: the front matter is flat scalars, and a
    parser that accepts exactly what we write fails loudly on a typo instead of
    silently coercing it into something unexpected.
    """
    match = _FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"{fallback_id}: missing front matter")

    fields: dict[str, object] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key not in _META_FIELDS:
            raise ValueError(f"{fallback_id}: unknown front matter key {key!r}")
        fields[key] = value or None

    if fields.get("effective_date"):
        fields["effective_date"] = date.fromisoformat(str(fields["effective_date"]))
    fields.setdefault("doc_id", fallback_id)

    return DocMeta(**fields), raw[match.end() :]  # type: ignore[arg-type]


def load_handwritten(directory: Path = KB_DIR) -> list[Document]:
    documents = []
    for path in sorted(directory.glob("*.md")):
        meta, body = _parse_front_matter(
            path.read_text(encoding="utf-8"), path.stem
        )
        documents.append(Document(meta=meta, text=body))
    return documents


# --- generated: product factsheets -----------------------------------------

_OBJECTIVE = {
    "equity": "seeks long-term capital growth through exposure to listed equity",
    "reit": "seeks income and moderate capital growth from a portfolio of income-producing real estate",
    "bond": "seeks income and capital preservation from a portfolio of debt securities",
    "commodity": "seeks to track the spot price of the underlying commodity less expenses",
}

_RISK_NOTES = {
    "equity": [
        "Equity prices can fall sharply and without warning. A peak-to-trough decline of a third has occurred repeatedly in developed markets and should be treated as a normal feature rather than a tail event.",
        "Concentration risk applies where the position is large relative to the portfolio. See the concentration limits policy for the applicable thresholds.",
    ],
    "reit": [
        "Distributions are not guaranteed and depend on occupancy, rental reversion and the cost of debt. Rising interest rates raise financing costs and simultaneously reduce the present value of future distributions, so REIT prices are more rate-sensitive than their yield suggests.",
        "Singapore REITs are correlated with each other, with the local property cycle and with Singapore banks. Holding several is less diversifying than the line-item count implies.",
    ],
    "bond": [
        "Bond prices fall when yields rise. Duration measures that sensitivity: a fund with a duration of six years loses roughly 6% of value for a one percentage point rise in yields.",
        "Credit risk applies to corporate holdings. Government-issued paper carries interest rate risk but negligible credit risk.",
    ],
    "commodity": [
        "The holding produces no income and no earnings. Its return is entirely price change, and it can go many years without a positive one.",
        "It is held for its low correlation with equity in stress periods, not for expected return, and should be sized accordingly.",
    ],
}

_TAX_NOTES = {
    "SGD": "Singapore does not tax capital gains. Singapore-sourced dividends are received without further tax in the hands of a resident individual under the one-tier system.",
    "USD": "US-domiciled funds withhold 30% on dividends paid to a Singapore resident, and US situs assets may fall within US estate tax above the non-resident exemption. An Ireland-domiciled equivalent reduces withholding on US dividends to 15% and avoids US estate tax exposure.",
}

_CPFIS_NOTE = {
    True: "Eligible under CPFIS-OA subject to the 35% limit on shares, property funds and corporate bonds. Any CPFIS purchase must be justified against the 2.5% Ordinary Account floor it displaces.",
    False: "Not available under the CPF Investment Scheme. Purchases must be funded from cash or SRS.",
}


def _factsheet(instrument: Instrument, rng: random.Random) -> Document:
    cpfis = instrument.currency == "SGD" and instrument.asset_class != "commodity"
    yield_pct = f"{instrument.annual_yield * 100:.2f}%"
    fee = rng.choice([0.03, 0.07, 0.09, 0.15, 0.20, 0.40])

    body = f"""
# {instrument.name} ({instrument.ticker})

{instrument.name} {_OBJECTIVE[instrument.asset_class]} in {instrument.region}. The instrument is denominated in {instrument.currency} and trades on its primary listing venue in that currency.

Asset class: {instrument.asset_class}. Region: {instrument.region}. Reporting currency: {instrument.currency}. Indicative distribution yield: {yield_pct} per annum. Estimated total expense ratio: {fee:.2f}% per annum.

# Risk

{_RISK_NOTES[instrument.asset_class][0]}

{_RISK_NOTES[instrument.asset_class][1]}

Currency risk applies to any holding denominated other than in SGD. A client whose liabilities and goals are in SGD carries the full {instrument.currency} exposure of this position unless it is hedged, and for goals inside three years the currency policy requires that it is not held unhedged.

# Costs and tax

The total expense ratio of {fee:.2f}% is deducted from fund assets and is not billed separately. Brokerage, custody and any foreign exchange spread on conversion into {instrument.currency} apply in addition and are disclosed separately.

{_TAX_NOTES.get(instrument.currency, _TAX_NOTES["SGD"])}

# Account eligibility

{_CPFIS_NOTE[cpfis]}

SRS funds may be used to purchase this instrument where it is available on the platform, without the percentage limits that apply under CPFIS. Because SRS money is locked until statutory retirement age, its horizon is long and an allocation to cash within SRS is a mismatch.

# Suitability

This instrument is classified as an Excluded Investment Product and does not require a Customer Knowledge Assessment. Suitability nevertheless requires a documented reasonable basis: the client's objectives, horizon, tolerance and existing portfolio must support the position and its size.
""".strip()

    return Document(
        meta=DocMeta(
            doc_id=f"factsheet-{instrument.ticker.lower().replace('.', '-')}",
            title=f"Factsheet: {instrument.name} ({instrument.ticker})",
            doc_type="factsheet",
            authority="issuer",
            effective_date=date(2025, 10, 1),
            ticker=instrument.ticker,
            asset_class=instrument.asset_class,
            region=instrument.region,
        ),
        text=body,
    )


# --- generated: market commentary ------------------------------------------

_DIRECTION = ["advanced", "retreated", "was little changed", "outperformed", "lagged"]
_DRIVERS = [
    "the path of policy rates",
    "the earnings revision cycle",
    "currency moves against the SGD",
    "the term premium on long-dated government paper",
    "foreign investor flows",
    "the credit impulse in China",
    "energy and freight costs",
    "the pace of disinflation in services",
]
_RISKS = [
    "a re-acceleration in services inflation forcing policy tighter for longer",
    "a sharper slowdown in external demand than currently discounted",
    "refinancing pressure on borrowers who fixed at much lower rates",
    "concentration of index returns in a small number of very large constituents",
    "a disorderly move in the currency that overwhelms local fundamentals",
    "valuation multiples that already assume the easing cycle arrives on schedule",
]
_POSITION = [
    "We remain at model weight and see no case for a tactical deviation.",
    "We are comfortable holding through the current volatility and would add on weakness toward the lower end of the range.",
    "We would fund any addition from cash rather than by reducing bonds.",
    "We would trim positions that have drifted more than ten points above their model weight.",
    "We prefer broad exposure to single-name selection in this segment.",
]


def _commentary(
    asset_class: str, region: str, quarter: date, rng: random.Random
) -> Document:
    q = (quarter.month - 1) // 3 + 1
    label = f"Q{q} {quarter.year}"
    move = rng.uniform(-9.5, 12.5)
    drivers = rng.sample(_DRIVERS, 2)
    risks = rng.sample(_RISKS, 2)

    body = f"""
# {region} {asset_class} - {label} review

{region} {asset_class} {rng.choice(_DIRECTION)} over {label}, returning {move:+.1f}% in local currency terms over the quarter. Translated into SGD the return differed by the currency move over the period, which for clients measuring in SGD is the number that matters.

The quarter was driven principally by {drivers[0]} and secondarily by {drivers[1]}. Neither is a permanent feature and both are already partly reflected in current pricing, which is why we treat quarterly attribution as explanation rather than as a forecast input.

# Valuation

Valuations across {region} {asset_class} sit within their long-run range rather than at an extreme. Dispersion within the segment is wider than the headline suggests, and the aggregate multiple is less informative than usual as a result.

We do not publish a target level or an expected return for the coming quarter. Positioning is set from the client's horizon and mandate, not from a quarterly view.

# Risks

The two risks we watch most closely are {risks[0]}, and {risks[1]}.

A client whose goal funded from this exposure falls inside three years should not be carrying it unhedged, irrespective of the outlook described here. Horizon governs, and the currency and liquidity policies apply regardless of the view.

# Positioning

{rng.choice(_POSITION)} {rng.choice(_POSITION)}

This commentary is general and is not a recommendation for any client. Any action must be assessed against the individual client's model portfolio, drift, concentration limits and goal funding position before it is proposed.
""".strip()

    return Document(
        meta=DocMeta(
            doc_id=f"commentary-{asset_class}-{region.lower().replace(' ', '-')}-{quarter.year}q{q}",
            title=f"{region} {asset_class} commentary, {label}",
            doc_type="commentary",
            authority="internal",
            effective_date=quarter,
            asset_class=asset_class,
            region=region,
        ),
        text=body,
    )


# --- generated: client meeting notes ---------------------------------------

_TOPICS = [
    "reviewed the current allocation against the model portfolio and confirmed the risk profile still reflects the client's circumstances",
    "discussed goal funding and whether the stated target amounts remain realistic",
    "walked through the CPF position, the interest floors on each account and what changes at 55",
    "covered the SRS balance, the contribution cap and the tax relief actually achieved this year",
    "went through the mortgage, the current rate and whether prepayment is preferable to investing the surplus",
    "reviewed the emergency reserve against six months of stated expenses",
    "discussed currency exposure and the mismatch between SGD goals and USD-denominated holdings",
    "reviewed concentration in Singapore financials and REITs and the correlation between them",
]
_CLIENT_VIEWS = [
    "The client is comfortable with the current level of volatility and confirmed they held through the last drawdown without selling.",
    "The client expressed discomfort with the size of the largest single position and asked what a reduction would cost.",
    "The client wants to prioritise the nearest goal even at the cost of long-term return.",
    "The client would prefer not to sell anything at a loss and asked whether new contributions could do the work instead.",
    "The client asked to revisit the risk profile questionnaire, saying their circumstances have changed since it was completed.",
    "The client raised a possible property purchase in the next few years and asked what it would require.",
]
_ACTIONS = [
    "Agreed to model the effect of directing the monthly surplus to the nearest goal before the next review.",
    "Agreed to prepare a reduction proposal for the largest single position, funded into the model weights.",
    "Agreed to deploy the idle SRS cash rather than leave it earning the account rate.",
    "Agreed to check relief headroom against the S$80,000 cap before any further SRS contribution.",
    "No action agreed. The client asked to revisit at the next scheduled review.",
    "Agreed to reassess the risk profile at the next meeting, as the current one is more than two years old.",
]


def _meeting_note(client: Client, index: int, rng: random.Random) -> Document:
    when = date(2025, 12, 1)
    months_back = index * 3 + 1
    year = when.year - (months_back // 12)
    month = when.month - (months_back % 12)
    if month <= 0:
        month += 12
        year -= 1
    met = date(year, month, min(rng.randint(3, 26), 28))

    topics = rng.sample(_TOPICS, 3)

    body = f"""
# Meeting note - {client.name} ({client.client_id}) - {met.isoformat()}

Present: the client and their adviser. Review of the {client.risk_profile} mandate for a {client.segment} client aged {client.age}.

We {topics[0]}. We also {topics[1]}, and {topics[2]}.

# Client view

{rng.choice(_CLIENT_VIEWS)} {rng.choice(_CLIENT_VIEWS)}

The client restated that their priority ordering across goals has not changed since the last review, and asked that any proposal respect it.

# Actions

{rng.choice(_ACTIONS)} {rng.choice(_ACTIONS)}

Nothing in this note constitutes an instruction to transact. Any trade arising requires separate confirmation from the client.
""".strip()

    return Document(
        meta=DocMeta(
            doc_id=f"meeting-{client.client_id.lower()}-{met.isoformat()}",
            title=f"Meeting note, {client.name}, {met.isoformat()}",
            doc_type="meeting_note",
            authority="internal",
            effective_date=met,
            client_id=client.client_id,
            risk_profiles=(client.risk_profile,),
        ),
        text=body,
    )


# --- generated: FAQs -------------------------------------------------------

_FAQS = [
    ("How much can I contribute to SRS this year?",
     "The annual cap is S$15,300 for Singapore Citizens and Permanent Residents and S$35,700 for foreigners. Contributions must reach the account by 31 December to count toward that year's relief, and the relief itself sits under the overall S$80,000 personal relief cap."),
    ("What interest does my CPF earn?",
     "The Ordinary Account earns a floor of 2.5% per annum and the Special, MediSave and Retirement Accounts earn 4.0%. An extra 1% applies on the first S$60,000 of combined balances, with at most S$20,000 counted from the Ordinary Account, and a further 1% on the first S$30,000 from age 55."),
    ("What happens to my Special Account at 55?",
     "From January 2025 the Special Account is closed at 55. Savings move to the Retirement Account up to the Full Retirement Sum, and anything above that goes to the Ordinary Account where it earns 2.5% rather than 4.0%."),
    ("Should I pay down my mortgage or invest?",
     "Prepaying an HDB loan at 2.6% is a guaranteed, tax-free 2.6% return. Investing must beat that after charges, with volatility you can tolerate. Neither answer is universal: it depends on your emergency reserve, your horizon and whether the money is needed for a nearer goal."),
    ("How much stamp duty will a second property cost me?",
     "A Singapore Citizen pays 20% Additional Buyer's Stamp Duty on a second residential property, on top of Buyer's Stamp Duty on the tiered scale. None of it can be financed by the mortgage, so it must be funded in cash."),
    ("Why is my portfolio different from the model?",
     "Prices move, so weights drift away from their targets. Drift within 5 percentage points needs no action, 5 to 10 points is rebalanced at the next review, and beyond 10 points triggers a proposal within ten business days."),
    ("Can I use CPF to buy shares?",
     "Ordinary Account savings above the first S$20,000 may be invested under CPFIS-OA, with shares, property funds and corporate bonds capped at 35% of investible savings and gold at 10%. The return has to beat the 2.5% floor those savings would otherwise earn."),
    ("Is my emergency fund big enough?",
     "The policy is six months of expenses held in cash. CPF and SRS do not count, because withdrawal is restricted by scheme rules rather than by market conditions. Where the reserve is short, rebuilding it takes priority over every other recommendation."),
    ("What does unrealised gain mean?",
     "It is the difference between what a holding is worth now and what you paid for it, before you sell. Singapore does not tax capital gains, so realising it does not create a tax charge, but transaction costs and the spread still apply."),
    ("Why do you convert everything to SGD?",
     "Your goals and liabilities are in SGD, so that is the currency in which the answer has to be true. A USD holding that rose 5% while USD fell 6% against SGD has lost you money in the terms that matter."),
]

_FAQ_TRANSLATIONS = {
    "zh": ("常见问题", "以下答复以已公布的规则为依据，并不构成针对个别客户的建议。"),
    "ms": ("Soalan Lazim", "Jawapan ini berdasarkan peraturan yang diterbitkan dan bukan nasihat khusus untuk mana-mana pelanggan."),
    "ta": ("அடிக்கடி கேட்கப்படும் கேள்விகள்", "இந்த பதில்கள் வெளியிடப்பட்ட விதிகளை அடிப்படையாகக் கொண்டவை; இவை எந்தவொரு வாடிக்கையாளருக்கும் குறிப்பிட்ட ஆலோசனை அல்ல."),
}


def _faqs() -> list[Document]:
    documents = []
    for i, (question, answer) in enumerate(_FAQS):
        documents.append(
            Document(
                meta=DocMeta(
                    doc_id=f"faq-{i:02d}",
                    title=f"FAQ: {question}",
                    doc_type="faq",
                    authority="internal",
                    effective_date=date(2025, 9, 1),
                ),
                text=f"# {question}\n\n{answer}\n\nThis answer describes the published rules and is not specific advice. Your own position may change the conclusion.",
            )
        )
        # A short multilingual gloss per FAQ. Enough for cross-language
        # retrieval to be demonstrable without pretending to be a translation
        # of record.
        for language, (heading, disclaimer) in _FAQ_TRANSLATIONS.items():
            documents.append(
                Document(
                    meta=DocMeta(
                        doc_id=f"faq-{i:02d}-{language}",
                        title=f"{heading}: {question}",
                        doc_type="faq",
                        authority="internal",
                        language=language,
                        effective_date=date(2025, 9, 1),
                    ),
                    text=f"# {heading}\n\n{question}\n\n{answer}\n\n{disclaimer}",
                )
            )
    return documents


# --- assembly --------------------------------------------------------------

def _quarters(count: int, end: date = TODAY) -> list[date]:
    out = []
    year, month = end.year, ((end.month - 1) // 3) * 3 + 1
    for _ in range(count):
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
        out.append(date(year, month, 1))
    return out


def build_corpus() -> list[Document]:
    """Every document in the knowledge base, in a stable order."""
    documents = load_handwritten()

    rng = random.Random(CORPUS_SEED)
    for instrument in UNIVERSE.values():
        documents.append(_factsheet(instrument, rng))

    segments = sorted({(i.asset_class, i.region) for i in UNIVERSE.values()})
    for asset_class, region in segments:
        for quarter in _quarters(28):
            documents.append(_commentary(asset_class, region, quarter, rng))

    for client in BOOK.values():
        for index in range(12):
            documents.append(_meeting_note(client, index, rng))

    documents.extend(_faqs())

    seen = {d.meta.doc_id for d in documents}
    if len(seen) != len(documents):
        raise ValueError("duplicate doc_id in corpus")
    return documents


@lru_cache(maxsize=1)
def build_chunks() -> tuple[Chunk, ...]:
    """The corpus flattened into chunks. Cached: the build is pure."""
    chunks: list[Chunk] = []
    for document in build_corpus():
        chunks.extend(chunk_document(document))
    return tuple(chunks)
