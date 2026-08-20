"""SIC → GICS-sector mapping — pure, no I/O.

Why SIC at all
--------------
``instruments.sector`` is load-bearing: ``rel_strength`` (both the detector
and the feature), ``rs_20d_sector``, ``ps_sector_pctile``, the deep-dive
sector buckets, and the risk engine's sector caps all key off it. It was
populated for 40 of 11,505 rows — the hand-written ``STARTER_SECTORS``
table — so all of that was dead for 99.7% of the market.

GICS itself is licensed (S&P/MSCI); there is no free, redistributable
symbol→GICS feed. SIC is the free alternative the SEC already publishes for
every filer, and the SEC's own classification is what "the market" uses to
route filings. This module maps it onto the eleven GICS sector NAMES the
rest of the platform already speaks (see ``default_sector_etf_map``), so
downstream code needs no change.

Fidelity, stated honestly
-------------------------
This is an approximation, not GICS. SIC predates the 2018 GICS
restructuring that created Communication Services and the 2016 one that
split out Real Estate, and its manufacturing codes cut along production
process rather than end market. The mapping below handles the cases where
that difference actually matters — internet/media into Communication
Services, REITs into Real Estate, pipelines into Energy rather than
Utilities, pharma/biotech/devices into Health Care, semis and software into
Information Technology, soap/cosmetics into Consumer Staples rather than
Materials — and accepts coarser results elsewhere. A name whose SIC has no
confident sector reading gets NO sector rather than a wrong one: an absent
sector degrades a feature to NaN, a wrong one silently corrupts a
sector-relative comparison and a risk cap.

Rules are evaluated most-specific-first: exact 4-digit codes, then narrow
ranges, then major groups.
"""

from __future__ import annotations

# --- GICS sector names, exactly as the rest of the platform spells them ---
ENERGY = "Energy"
MATERIALS = "Materials"
INDUSTRIALS = "Industrials"
CONSUMER_DISCRETIONARY = "Consumer Discretionary"
CONSUMER_STAPLES = "Consumer Staples"
HEALTH_CARE = "Health Care"
FINANCIALS = "Financials"
INFORMATION_TECHNOLOGY = "Information Technology"
COMMUNICATION_SERVICES = "Communication Services"
UTILITIES = "Utilities"
REAL_ESTATE = "Real Estate"

#: (low, high, sector) inclusive SIC ranges, most specific FIRST. The first
#: matching rule wins, so a 4-digit exception may precede the broad range it
#: sits inside.
_SECTOR_RULES: tuple[tuple[int, int, str], ...] = (
    # --- exceptions that must beat their enclosing ranges ----------------
    # Soap, cosmetics, household products: SIC files them under Chemicals,
    # GICS calls them Household & Personal Products.
    (2840, 2844, CONSUMER_STAPLES),
    # Pharmaceutical preparations, biologics, diagnostics: inside Chemicals.
    (2833, 2836, HEALTH_CARE),
    # Drugs wholesale sits inside nondurable wholesale.
    (5122, 5122, HEALTH_CARE),
    # Medical/hospital equipment wholesale sits inside durable wholesale.
    (5047, 5047, HEALTH_CARE),
    # Petroleum wholesale sits inside nondurable wholesale.
    (5171, 5172, ENERGY),
    # Groceries wholesale -> Food Distributors.
    (5140, 5149, CONSUMER_STAPLES),
    # Drug stores sit inside miscellaneous retail.
    (5912, 5912, CONSUMER_STAPLES),
    # Eating places sit inside retail; GICS calls them Restaurants.
    (5810, 5813, CONSUMER_DISCRETIONARY),
    # Motor vehicles/parts sit inside Transportation Equipment (Industrials).
    (3711, 3716, CONSUMER_DISCRETIONARY),
    (3751, 3751, CONSUMER_DISCRETIONARY),  # motorcycles & bicycles
    # Household appliances sit inside Electrical Equipment.
    (3630, 3639, CONSUMER_DISCRETIONARY),
    # Lab/analytical and medical instruments sit inside Instruments.
    (3826, 3827, HEALTH_CARE),  # lab analytical / lab apparatus
    (3841, 3851, HEALTH_CARE),  # surgical, medical, dental, ophthalmic
    # Advertising sits inside Business Services; GICS: Media.
    (7310, 7319, COMMUNICATION_SERVICES),
    # Computer programming, software, data processing sit inside Business
    # Services; GICS: Software & IT Services.
    (7370, 7379, INFORMATION_TECHNOLOGY),
    # REITs and real-estate operators sit inside Holding/Investment Offices.
    (6798, 6798, REAL_ESTATE),
    # Blank checks (SPACs) — a financing vehicle, not an operating company.
    (6770, 6770, FINANCIALS),
    # Commercial physical & biological research: where most clinical-stage
    # biotech files. Inside "Engineering/Research/Management" (Industrials).
    (8731, 8731, HEALTH_CARE),
    # --- Energy -----------------------------------------------------------
    (1200, 1299, ENERGY),  # coal — GICS Consumable Fuels
    (1300, 1399, ENERGY),  # crude petroleum, gas, oilfield services
    (2911, 2911, ENERGY),  # petroleum refining
    (4610, 4613, ENERGY),  # pipelines except natural gas
    # --- Materials --------------------------------------------------------
    (1000, 1099, MATERIALS),  # metal mining
    (1400, 1499, MATERIALS),  # nonmetallic minerals
    (2200, 2299, MATERIALS),  # textile mill products
    (2400, 2499, MATERIALS),  # lumber & wood
    (2600, 2699, MATERIALS),  # paper
    (2800, 2899, MATERIALS),  # chemicals (exceptions handled above)
    (3000, 3089, MATERIALS),  # rubber & plastics materials
    (3200, 3299, MATERIALS),  # stone, clay, glass, concrete
    (3300, 3399, MATERIALS),  # primary metals
    # --- Industrials ------------------------------------------------------
    (1500, 1799, INDUSTRIALS),  # construction & engineering
    (3400, 3499, INDUSTRIALS),  # fabricated metal
    (3500, 3569, INDUSTRIALS),  # industrial machinery
    (3580, 3599, INDUSTRIALS),  # service & general industrial machinery
    (3600, 3629, INDUSTRIALS),  # electrical equipment & components
    (3700, 3799, INDUSTRIALS),  # aerospace, rail, marine (autos above)
    (3812, 3812, INDUSTRIALS),  # search/navigation — aerospace & defense
    (4000, 4099, INDUSTRIALS),  # railroads
    (4100, 4199, INDUSTRIALS),  # transit & ground passenger
    (4200, 4299, INDUSTRIALS),  # trucking & logistics
    (4400, 4499, INDUSTRIALS),  # marine
    (4500, 4599, INDUSTRIALS),  # airlines & air freight
    (4700, 4789, INDUSTRIALS),  # transportation services
    (7320, 7369, INDUSTRIALS),  # credit reporting, staffing, facilities
    (8711, 8713, INDUSTRIALS),  # engineering, architectural, surveying
    (8740, 8748, INDUSTRIALS),  # management & consulting services
    # --- Consumer Discretionary ------------------------------------------
    (2300, 2399, CONSUMER_DISCRETIONARY),  # apparel
    (2500, 2599, CONSUMER_DISCRETIONARY),  # furniture & fixtures
    (3100, 3199, CONSUMER_DISCRETIONARY),  # leather & footwear
    (3900, 3999, CONSUMER_DISCRETIONARY),  # jewelry, toys, sporting goods
    (5200, 5399, CONSUMER_DISCRETIONARY),  # building supply, general merch
    (5500, 5599, CONSUMER_DISCRETIONARY),  # auto dealers
    (5600, 5799, CONSUMER_DISCRETIONARY),  # apparel/home/electronics retail
    (5900, 5999, CONSUMER_DISCRETIONARY),  # misc retail (drug stores above)
    (7000, 7019, CONSUMER_DISCRETIONARY),  # hotels & lodging
    (7200, 7299, CONSUMER_DISCRETIONARY),  # personal & consumer services
    (7500, 7549, CONSUMER_DISCRETIONARY),  # automotive services
    (7900, 7999, CONSUMER_DISCRETIONARY),  # leisure facilities & products
    (8200, 8299, CONSUMER_DISCRETIONARY),  # education services
    # --- Consumer Staples -------------------------------------------------
    (100, 299, CONSUMER_STAPLES),  # agricultural production
    (2000, 2199, CONSUMER_STAPLES),  # food, beverages, tobacco
    (5400, 5499, CONSUMER_STAPLES),  # food & staples retailing
    # --- Health Care ------------------------------------------------------
    (8000, 8099, HEALTH_CARE),  # health services, hospitals, managed care
    # --- Financials -------------------------------------------------------
    (6000, 6299, FINANCIALS),  # banks, credit, brokers, exchanges
    (6300, 6499, FINANCIALS),  # insurance carriers, agents
    (6700, 6799, FINANCIALS),  # investment offices (REITs/SPACs above)
    # --- Real Estate ------------------------------------------------------
    (6500, 6599, REAL_ESTATE),  # real estate operators, agents, developers
    # --- Information Technology -------------------------------------------
    (3570, 3579, INFORMATION_TECHNOLOGY),  # computer & office equipment
    (3660, 3699, INFORMATION_TECHNOLOGY),  # comms equipment, semis, components
    (3820, 3825, INFORMATION_TECHNOLOGY),  # measuring/controlling instruments
    (3860, 3873, INFORMATION_TECHNOLOGY),  # photographic, optical, watches
    # --- Communication Services -------------------------------------------
    (2700, 2799, COMMUNICATION_SERVICES),  # publishing
    (4800, 4899, COMMUNICATION_SERVICES),  # telecom, broadcasting, cable
    (7800, 7841, COMMUNICATION_SERVICES),  # motion pictures & entertainment
    # --- Utilities --------------------------------------------------------
    (4900, 4991, UTILITIES),  # electric, gas, water, sanitary
)

#: SIC major group (first two digits) → label, used for
#: ``instruments.industry``. Purely descriptive; nothing computes on it.
_MAJOR_GROUPS: dict[int, str] = {
    1: "Agricultural Production - Crops",
    2: "Agricultural Production - Livestock",
    7: "Agricultural Services",
    8: "Forestry",
    9: "Fishing, Hunting and Trapping",
    10: "Metal Mining",
    12: "Coal Mining",
    13: "Oil and Gas Extraction",
    14: "Nonmetallic Minerals, Except Fuels",
    15: "Building Construction - General Contractors",
    16: "Heavy Construction, Except Building",
    17: "Construction - Special Trade Contractors",
    20: "Food and Kindred Products",
    21: "Tobacco Products",
    22: "Textile Mill Products",
    23: "Apparel and Other Finished Products",
    24: "Lumber and Wood Products",
    25: "Furniture and Fixtures",
    26: "Paper and Allied Products",
    27: "Printing, Publishing and Allied Industries",
    28: "Chemicals and Allied Products",
    29: "Petroleum Refining and Related Industries",
    30: "Rubber and Miscellaneous Plastics Products",
    31: "Leather and Leather Products",
    32: "Stone, Clay, Glass and Concrete Products",
    33: "Primary Metal Industries",
    34: "Fabricated Metal Products",
    35: "Industrial Machinery and Computer Equipment",
    36: "Electronic and Other Electrical Equipment",
    37: "Transportation Equipment",
    38: "Measuring, Analyzing and Controlling Instruments",
    39: "Miscellaneous Manufacturing Industries",
    40: "Railroad Transportation",
    41: "Local and Suburban Transit",
    42: "Motor Freight Transportation and Warehousing",
    44: "Water Transportation",
    45: "Transportation by Air",
    46: "Pipelines, Except Natural Gas",
    47: "Transportation Services",
    48: "Communications",
    49: "Electric, Gas and Sanitary Services",
    50: "Wholesale Trade - Durable Goods",
    51: "Wholesale Trade - Nondurable Goods",
    52: "Building Materials and Garden Supplies",
    53: "General Merchandise Stores",
    54: "Food Stores",
    55: "Automotive Dealers and Service Stations",
    56: "Apparel and Accessory Stores",
    57: "Home Furniture and Furnishings Stores",
    58: "Eating and Drinking Places",
    59: "Miscellaneous Retail",
    60: "Depository Institutions",
    61: "Nondepository Credit Institutions",
    62: "Security and Commodity Brokers and Exchanges",
    63: "Insurance Carriers",
    64: "Insurance Agents, Brokers and Service",
    65: "Real Estate",
    67: "Holding and Other Investment Offices",
    70: "Hotels and Other Lodging Places",
    72: "Personal Services",
    73: "Business Services",
    75: "Automotive Repair, Services and Parking",
    76: "Miscellaneous Repair Services",
    78: "Motion Pictures",
    79: "Amusement and Recreation Services",
    80: "Health Services",
    81: "Legal Services",
    82: "Educational Services",
    83: "Social Services",
    84: "Museums and Botanical/Zoological Gardens",
    86: "Membership Organizations",
    87: "Engineering, Accounting, Research and Management",
    89: "Services, Not Elsewhere Classified",
    99: "Nonclassifiable Establishments",
}


def normalize_sic(sic: str | int | None) -> int | None:
    """Parse a SIC code; ``None`` for blank/placeholder/unparseable values."""
    if sic is None:
        return None
    text = str(sic).strip()
    if not text or not text.isdigit():
        return None
    code = int(text)
    # 0000 and the 9995 "non-classifiable" placeholder carry no information.
    if code <= 0 or code == 9995:
        return None
    return code


def sector_for_sic(sic: str | int | None) -> str | None:
    """GICS-sector name for a SIC code, or ``None`` when unmappable.

    Returning ``None`` is deliberate. A missing sector makes a
    sector-relative feature abstain; a wrong sector makes it lie.
    """
    code = normalize_sic(sic)
    if code is None:
        return None
    for low, high, sector in _SECTOR_RULES:
        if low <= code <= high:
            return sector
    return None


def industry_for_sic(sic: str | int | None) -> str | None:
    """SIC major-group label for ``instruments.industry`` (descriptive only)."""
    code = normalize_sic(sic)
    if code is None:
        return None
    return _MAJOR_GROUPS.get(code // 100)
