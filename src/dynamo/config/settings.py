"""Platform configuration. Design: doc 01 section 5 (environments), doc 00 section 6 (budget)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dynamo.domain.symbols import is_crypto_symbol


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Repository root: the nearest ancestor of this file holding pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


class DynamoEnv(StrEnum):
    DEV = "DEV"
    TEST = "TEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    dynamo_env: DynamoEnv = Field(default=DynamoEnv.DEV, alias="DYNAMO_ENV")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://dynamo:change-me@localhost:5432/dynamo_dev",
        alias="DATABASE_URL",
    )

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_monthly_budget_usd: float = Field(default=90.0, alias="CLAUDE_MONTHLY_BUDGET_USD")
    # Tier models: Sonnet 5 (intro $2/$10 through 2026-08-31, then $3/$15) is
    # cheaper AND newer than Sonnet 4.6; Opus 5 matches Opus 4.6 pricing with a
    # far lower prompt-cache minimum (512 vs 4096 tokens).
    model_screen: str = "claude-haiku-4-5"
    model_analyst: str = "claude-sonnet-5"
    model_arbiter: str = "claude-opus-5"

    # --- Discovery funnel throughput (doc 02 §2, doc 03 §1) ---------------
    # These five numbers are ONE budget, not five independent knobs. The
    # constraint, and the reasoning behind each default, is in
    # dynamo.research.funnel; test_funnel.py asserts the shipped defaults
    # satisfy it, so changing one of them in isolation fails the suite.
    #
    #   triage_cap * material_rate  <=  deep_dive_promotion_slots
    #   16 * 0.30 = 4.8  <=  5.   (16 is the largest cap that fits)
    #
    # Before: triage 60/night x ~30% material = ~18 promotions/night against
    # 8 deep dives, all 8 claimable by promotions. The queue grew ~10/night,
    # ~540 names sat in Tier 3 against a 30-day window, ~44% of promotions
    # expired unanalyzed, and Tier 1 got zero dives on any night with 8+
    # promotions — i.e. every night.
    deep_dive_cap_per_night: int = 8
    # Of those 8, at most this many go to brand-new promotions. The other 3
    # cover events and staleness on the existing universe: ~55 Tier-1 names
    # against a 14-day staleness trigger need ~4/night to stay fresh, so 3
    # is a deliberate (documented) under-provision in favor of discovery.
    # Unused promotion slots are NOT wasted — they fall through to events
    # and staleness, and unused event/stale slots fall back to promotions.
    deep_dive_promotion_slots: int = 5
    # Observed rate at which Haiku triage returns "material". Not a target:
    # if the real rate moves, this is the number to update, and the funnel
    # test will then tell you the triage cap no longer fits.
    triage_material_rate: float = 0.30
    tier2_triage_cap_per_night: int = 16
    # Sessions a symbol sits out after being triaged. Shorter than the
    # deep-dive slate's 5 because triage is cheap and its job is breadth;
    # not 0 because a range_break flag persists for days and would
    # otherwise re-consume a slot every night.
    tier2_triage_cooldown_sessions: int = 3
    # Tier-3 promotion window. NOT a throughput lever — with arrivals under
    # the service rate the queue is bounded by one night's promotions, so
    # this governs how long a name keeps full quant treatment after its
    # dive, nothing more.
    tier3_promotion_window_days: int = 30

    # --- Research backend selection (ADR-0014) ---
    # "api" (default): AnthropicClient — per-token billing, Message Batches at
    #   50% off, prompt caching, effectively unbounded scale.
    # "claude_code": route committee calls through the LOCAL Claude Code CLI,
    #   covered by the user's flat-rate subscription ($0 marginal cost for the
    #   nightly ~35-call paper-phase committee). Tradeoffs: subscription plan
    #   windows (5-hourly/weekly usage caps) can exhaust mid-run; no batch
    #   discount and no prompt caching (batches degrade to a sequential local
    #   loop); the scale ceiling is the plan limit — fine for nightly paper
    #   research, wrong for large replay campaigns. Personal-use automation
    #   within the user's own subscription limits.
    research_backend: str = Field(default="api", alias="RESEARCH_BACKEND")
    # Path to the `claude` executable; empty -> auto-discover ($PATH, then
    # known install locations — see claude_code_client.find_claude_binary).
    claude_code_binary: str = Field(default="", alias="CLAUDE_CODE_BINARY")
    # Tier models for the CLI backend. Fable 5 (training cutoff Jan 2026)
    # takes the arbiter seat — subscription-covered, so the strongest
    # available model costs the same as any other.
    claude_code_model_screen: str = "claude-haiku-4-5"
    claude_code_model_analyst: str = "claude-sonnet-5"
    claude_code_model_arbiter: str = "claude-fable-5"
    # When the plan window is exhausted mid-run: True -> fall through to the
    # Anthropic API per-call (requires ANTHROPIC_API_KEY, resumes per-token
    # billing); False (default) -> raise PlanLimitExhausted and stop until
    # the next window.
    claude_code_fallback_to_api: bool = False
    #: One-hop arbiter fallback when the arbiter model's own plan window is
    #: exhausted (Fable has a separate weekly allowance). Empty disables.
    claude_code_arbiter_fallback: str = "claude-opus-5"
    claude_code_timeout_seconds: int = 300
    #: Run the committee's bull and bear calls concurrently (eval Phase 4,
    #: docs/eval_tracing_2026-10-03.md). Off until the user opts in.
    committee_parallel_advocates: bool = False
    #: Candidate quant bundles scored nightly into shadow_quant_scores, beside
    #: production and never traded (comma-separated; empty disables).
    #: docs/preregistration_trend_shadow.md.
    shadow_quant_bundles: str = "trend-rule-v1"

    # --- Broker (paper first; live requires the gate, doc 09 section 7) ---
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )
    execution_auto_approve: bool = True  # paper default; always False in LIVE
    approval_ttl_minutes: int = 30
    reconciliation_cash_tolerance: float = 1.00
    #: Equity is compared with a RELATIVE tolerance: internal equity marks
    #: positions to the latest daily close, the broker to intraday prices, so a
    #: $1 band diverged on almost every run while any position was held. Cash
    #: and positions stay strict; they are what catch real bookkeeping errors.
    reconciliation_equity_tolerance_pct: float = 0.01

    # --- Crypto sleeve (ADR-0013; Alpaca crypto, paper-capable today) ---
    # Caps are deliberately conservative: crypto realized vol runs 3-5x
    # equities, so vol-parity sizing already shrinks crypto positions
    # naturally — the sleeve cap and per-position cap are belt-and-suspenders
    # hard bounds on top of that, enforced in the sizing cascade and the
    # constructor's compliance pass. Symbols use the canonical "BASE/QUOTE"
    # Alpaca pair form ("BTC/USD"); adapters normalize any slashless broker
    # form ("BTCUSD") at the edge (dynamo.domain.symbols).
    crypto_enabled: bool = Field(default=True, alias="CRYPTO_ENABLED")
    crypto_sleeve_cap: float = 0.10  # all crypto combined, fraction of equity
    crypto_max_position: float = 0.05  # any single pair, fraction of equity
    crypto_universe: tuple[str, ...] = (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "AVAX/USD",
        "LINK/USD",
    )

    # --- Idle-cash ballast (owner-requested: ON by default) ---
    # Cash above the true-cash floor defaults into a ballast ETF instead of
    # sitting in dollars; active positions are funded by trimming ballast and
    # exit proceeds flow back to it. The drawdown ladder and risk multipliers
    # govern ballast exposure like everything else (see
    # RuleBasedConstructor._ballast_lot). The DEFAULT is the diversified
    # index (SPY) on purpose: ballast bypasses per-position/sector caps, and
    # a single name carrying uncapped residual weight is not ballast — it is
    # a concentrated bet. A single-stock ballast is technically possible
    # (any non-crypto symbol validates) but deliberately not the default.
    ballast_enabled: bool = Field(default=True, alias="BALLAST_ENABLED")
    ballast_symbol: str = Field(default="SPY", alias="BALLAST_SYMBOL")

    # --- Market data ---
    tiingo_api_key: str = Field(default="", alias="TIINGO_API_KEY")
    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    edgar_user_agent: str = Field(
        default="Dynamo research (set EDGAR_USER_AGENT)", alias="EDGAR_USER_AGENT"
    )

    # --- Earnings-call transcripts (Alpha Vantage free key; defeatbeta backfill) ---
    alphavantage_api_key: str = Field(default="", alias="ALPHAVANTAGE_API_KEY")
    transcript_daily_cap: int = 20  # stay under Alpha Vantage's 25 requests/day free quota

    # --- TickerTick second news source (free, no key, 10 req/min) ---
    tickertick_enabled: bool = Field(default=True, alias="TICKERTICK_ENABLED")

    # --- Prediction markets (Kalshi tradable behind the gate; Polymarket reads keyless) ---
    kalshi_api_key_id: str = Field(default="", alias="KALSHI_API_KEY_ID")
    kalshi_private_key_path: str = Field(default="", alias="KALSHI_PRIVATE_KEY_PATH")
    pm_min_volume_usd: float = 25_000.0
    pm_max_spread: float = 0.03

    # --- OddsPipe cross-venue enrichment (ADR-0009 addendum; free tier) ---
    # Enrichment ONLY: the direct Kalshi/Polymarket adapters stay primary.
    # No key -> the OddsPipe adapter skips gracefully and nothing changes.
    oddspipe_api_key: str = Field(default="", alias="ODDSPIPE_API_KEY")
    oddspipe_min_spread: float = 0.03  # minimum YES spread fetched from /v1/spreads
    pm_cross_venue_veto_gap: float = 0.10  # |polymarket - estimated| veto threshold

    # --- Kalshi trading sleeve (ADR-0009; demo first, prod only behind the gate) ---
    # Base URLs exclude the /trade-api/v2 prefix (the adapters append it).
    kalshi_demo_base_url: str = Field(
        default="https://demo-api.kalshi.co", alias="KALSHI_DEMO_BASE_URL"
    )
    kalshi_prod_base_url: str = Field(
        default="https://api.elections.kalshi.com", alias="KALSHI_PROD_BASE_URL"
    )
    pm_sleeve_cap: float = 0.05  # sleeve total <= 5% of equity, hard
    pm_per_market_cap: float = 0.01  # per-market exposure <= 1% of equity
    pm_min_edge: float = 0.08  # |estimated - implied| floor
    pm_min_confidence: float = 0.6  # committee confidence floor
    pm_max_days_to_resolution: int = 90
    pm_kelly_fraction_cap: float = 0.25  # fractional Kelly, same cap as equities
    pm_max_open_markets: int = 10

    # --- Polymarket shadow-trading sleeve (ADR-0011; SIMULATED by default) ---
    # Polymarket has no demo exchange, so the sleeve simulates the broker
    # internally: decisions run through the same pm_* gates against the sim
    # bankroll and fills happen in our own tables at Gamma prices. This is
    # the ONLY behavior outside a fully-satisfied LIVE precondition chain.
    polymarket_sim_enabled: bool = True
    polymarket_sim_bankroll_usd: float = 1000.0
    polymarket_sim_slippage_cents: int = 1  # applied AGAINST us on every simulated fill

    # --- Polymarket LIVE execution (ADR-0012; gated, sim remains the default) ---
    # Real CLOB orders require ALL of: DYNAMO_ENV=LIVE, polymarket_live_enabled,
    # full credentials below, and a passing doc 09 §7 live gate (signed
    # approval + kill-switch absent + readiness). Anything missing falls back
    # to the sim broker with a loud warning — never the other way around.
    polymarket_api_key: str = Field(default="", alias="POLYMARKET_API_KEY")
    polymarket_api_secret: str = Field(default="", alias="POLYMARKET_API_SECRET")
    polymarket_passphrase: str = Field(default="", alias="POLYMARKET_PASSPHRASE")
    # Path to a file in secrets/ holding the Polygon wallet private key used
    # for EIP-712 order signing. Contents are NEVER logged or printed; the
    # broker reads the file lazily at client construction.
    polymarket_wallet_key_path: str = Field(default="", alias="POLYMARKET_WALLET_KEY_PATH")
    # Polymarket UI accounts trade through a proxy wallet: signature_type 1 =
    # email/Magic proxy, 2 = browser-wallet proxy, 0 = direct EOA. Proxy types
    # (1/2) also require the proxy ("funder") address that holds the USDC.
    polymarket_funder_address: str = Field(default="", alias="POLYMARKET_FUNDER_ADDRESS")
    polymarket_signature_type: int = 1
    polymarket_live_enabled: bool = False  # master switch, independent of DYNAMO_ENV
    polymarket_live_max_order_usd: float = 25.0  # hard per-order cap, enforced in the broker

    # --- Prediction-market arbitrage sleeve (SIM-ONLY; risk/pm_arbitrage) ---
    # There is deliberately NO live wiring for this sleeve: the pm_arbitrage
    # asset hard-codes the sim-ledger path and constructs no broker in any
    # environment (tested), so the enable flag is safe to default ON.
    pm_arbitrage_enabled: bool = True
    pm_arb_min_net_edge: float = 0.02  # locked-profit floor per $1 payout, after fees+slippage
    pm_arb_slippage: float = 0.01  # fill haircut vs the stored last/mid price, both finders
    pm_arb_sim_bankroll_usd: float = 1_000.0  # seed of the dedicated 'arbsim' sub-ledger
    pm_arb_max_stake_usd: float = 100.0  # per-candidate cost cap in the sim
    # Cross-venue pairs whose RESOLUTION CRITERIA a human has verified to
    # match, as "<KALSHI_TICKER>|<polymarket_id>" ids (the exact string the
    # asset logs for unverified candidates). EMPTY by default on purpose:
    # until a pair is verified, its candidates are research rows in
    # result.unverified — never sim trades. Verification steps are in the
    # pm_arbitrage_sim asset docstring (orchestration/assets/pm_trading.py).
    pm_arb_verified_pairs: tuple[str, ...] = ()
    # Title-matcher fallback (research/pm_matching): minimum similarity for a
    # Kalshi/Polymarket pair CANDIDATE when OddsPipe returns no catalog.
    # Candidates stay research rows until verified into pm_arb_verified_pairs.
    pm_match_min_score: float = 0.65

    # --- Favorites harvester sleeve (SIM-ONLY, GATED OFF; risk/pm_favorites) ---
    # pm_favorites_enabled stays False until the favorite-longshot calibration
    # study (python -m dynamo.cli.pm_calibration) demonstrates a POSITIVE NET
    # edge — realized YES rate above price + fees at the bottom of the 95% CI
    # — in the traded band. The study's printed verdict IS the gate: flip
    # this only on an "EDGE EXISTS" verdict, and set pm_fav_assumed_net_edge
    # to the demonstrated per-contract net edge at the same time (it defaults
    # to 0.0, which makes the selector emit nothing even when enabled — a
    # deliberate second fail-closed gate). Like the arbitrage sleeve, the
    # asset constructs no broker in any environment (tested).
    pm_favorites_enabled: bool = False
    pm_fav_min_price: float = 0.90  # buy-YES band floor (favorites only)
    pm_fav_max_price: float = 0.99  # buy-YES band ceiling
    pm_fav_assumed_net_edge: float = 0.0  # study-demonstrated $/contract; <= 0 selects nothing
    pm_fav_sim_bankroll_usd: float = 1_000.0  # seed of the dedicated 'favsim' sub-ledger
    pm_fav_max_stake_usd: float = 25.0  # per-candidate cost cap in the sim

    # --- Options sleeve (ADR-0010; defined-risk only, paper first) ---
    # Hard caps enforced in code (risk/options_sizing.py + domain validators):
    # the sleeve trades ONLY long calls, long puts, covered calls, and
    # cash-secured puts — unbounded-loss structures are unrepresentable.
    options_premium_at_risk_cap: float = 0.05  # long premium + assignment obligations
    options_per_position_cap: float = 0.01  # per-position premium at risk, of equity
    options_min_dte: int = 21
    options_max_dte: int = 90
    options_min_open_interest: int = 100
    options_max_spread_pct: float = 0.10  # (ask - bid) / mid ceiling
    options_min_conviction: float = 0.7
    options_iv_rank_max: float = 0.8  # never buy premium when IV rank is extreme
    options_max_open_positions: int = 8

    # --- Same-day tactical entries (15-min intraday loop) ---
    intraday_entry_cap_per_day: int = 3

    # --- Historical replay (backtest campaign Phase 4) ---
    # Safety margin added to each model's documented training-data cutoff when
    # the committee-replay knowledge gate decides whether a window is tainted:
    # refuse when window start <= cutoff + margin unless the run explicitly
    # acknowledges (--acknowledge-model-knowledge, recorded in config_json).
    replay_cutoff_margin_days: int = 30
    # Minimum asset-class-aware feature_coverage a symbol needs before replay
    # will rank it (campaign findings Phase 1). Below it the name abstains
    # instead of competing for a deep-dive slot off a mostly-empty feature
    # row. This is the platform default that composition roots pass into
    # ``ReplayConfig.min_feature_coverage`` (which is what actually
    # participates in the run's config_hash).
    replay_min_feature_coverage: float = 0.35
    # Preflight ship gate: refuse to start a replay that pins a quant bundle
    # whose training report says ships_ok=False on any task (campaign-v3 lost
    # to its own historic-average baseline on all three and still ran). Set
    # REPLAY_REQUIRE_SHIPPING_BUNDLE=false to downgrade the refusal to a loud
    # warning — e.g. to deliberately re-run a control experiment.
    replay_require_shipping_bundle: bool = True
    # Deep-dive shortlist diversity defaults (campaign findings item 19).
    # Platform defaults that composition roots pass into the matching
    # ``ReplayConfig.shortlist_*`` fields (which are what actually
    # participate in the run's config_hash). The control run's plain
    # top-4-by-score slate debated WMT on 70% of days, never once debated
    # AAPL/GOOGL/AMZN/MSFT, and gave crypto 29% of the slots on 9% of the
    # universe. Candidate band 0.6 (= top 40%) is deliberately WIDER than
    # the BUY percentile, which is unchanged: this widens what may be
    # DEBATED, never what counts as a BUY.
    replay_shortlist_candidate_percentile: float = 0.6
    replay_shortlist_crypto_slots: int = 1
    replay_shortlist_cooldown_sessions: int = 5
    replay_shortlist_rotation_slots: int = 1
    replay_shortlist_rotation_lookback_sessions: int = 20

    # --- Learning loop (doc 10; journal retrieval feeds dossier contexts) ---
    journal_retrieval_enabled: bool = True
    experience_retrieval_top_k: int = 3  # journal hits per symbol in the dossier
    # Journal vector search runs on sha256-seeded random vectors when the
    # optional ``embeddings`` extra is missing (dynamo.llm.embeddings). Those
    # hits are noise, and they enter live committee prompts labelled "similar
    # past situations", so retrieval ABSTAINS rather than degrading. Turning
    # this on restores the old behaviour and still logs a warning every run;
    # the real fix is `uv sync --extra embeddings && uv run python -m
    # dynamo.cli.journal_seed --reembed` (see docs/journal_embeddings.md).
    journal_allow_hash_fallback: bool = Field(
        default=False, alias="DYNAMO_JOURNAL_ALLOW_HASH_FALLBACK"
    )
    # Hierarchical (scope-chain) calibration, plan Phase 2. Gates the
    # hier-scope calibration_points emission AND the advisory calibrated-p
    # logging onto decision_experience. ADVISORY ONLY — nothing consumes
    # calibrated_probability for sizing.
    hierarchical_calibration_enabled: bool = True
    # --- Meta-model sizing consumption (plan Phase 3; ADVISORY-FIRST) ---
    # The nightly meta_scoring asset logs effective_probability rows to
    # meta_decision_scores whenever a PRODUCTION-stage meta-model exists in
    # the registry — that logging is always on and always advisory. THIS flag
    # gates the only consumption path: sizing tilts prefer the meta-model's
    # effective_probability over raw conviction. Flipping it to True requires
    # ALL of: a meta-model promoted to production through the full doc 08 §7
    # pipeline (candidate → validating → paper → shadow → production, signed
    # human approval at the gate), and a deliberate config change here.
    # Nothing in code ever flips it; False means the dark path stays dark and
    # sizing is bit-identical to the pre-meta platform.
    meta_sizing_enabled: bool = False

    # --- Social awareness (Reddit + StockTwits; detector input ONLY) ---
    # Social text is untrusted (pump-and-dump surface): it feeds the Tier-2
    # social_burst detector as an awareness trigger and never moves a score.
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="dynamo-research/0.1", alias="REDDIT_USER_AGENT")
    reddit_subreddits: str = Field(
        default="stocks,investing,wallstreetbets,StockMarket", alias="REDDIT_SUBREDDITS"
    )
    stocktwits_enabled: bool = Field(default=True, alias="STOCKTWITS_ENABLED")
    social_burst_zscore: float = 4.0
    social_min_mentions: int = 10
    social_lookback_days: int = 14

    # --- Risk limits (doc 06 §11: versioned, not source-edited) -----------
    # Path to a YAML/JSON file of RiskConfig field overrides. Relative paths
    # anchor to home_dir like the kill switch. Unset -> the documented
    # defaults in domain/models/risk.py, reported as version "default".
    # A path that does not exist is a REFUSAL, not a fallback: running on
    # defaults while believing an override is in force is the failure mode.
    risk_config_path: str = Field(default="", alias="DYNAMO_RISK_CONFIG")
    # doc 06 §2 risk bucket: low | medium | high (0.6x / 1.0x / 1.4x on the
    # per-name risk budget; never on gross_cap, cash_floor or any *_hard).
    risk_bucket: str = Field(default="", alias="DYNAMO_RISK_BUCKET")

    # --- Paths ---
    # Every relative path below is resolved against ``home_dir`` (see
    # :func:`dynamo_home`), NOT the process CWD. The kill switch is the reason:
    # the Dagster daemon, the API server and an operator's shell run from three
    # different working directories, and a CWD-relative "KILL_SWITCH" means
    # three different files — i.e. an operator can "engage" a switch that the
    # trading process never looks at.
    dynamo_home: str = Field(default="", alias="DYNAMO_HOME")
    parquet_dir: str = Field(default="data/parquet", alias="DYNAMO_PARQUET_DIR")
    artifact_dir: str = Field(default="artifacts", alias="DYNAMO_ARTIFACT_DIR")
    kill_switch_path: str = Field(default="KILL_SWITCH", alias="DYNAMO_KILL_SWITCH_PATH")
    approvals_dir: str = Field(default="approvals", alias="DYNAMO_APPROVALS_DIR")

    # --- Alerting (doc 12 §7) ---
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    # Severities at or above this are actually transmitted. Everything is
    # still written to the `alerts` table regardless — this only governs the
    # out-of-band page.
    notify_min_severity: str = Field(default="page", alias="DYNAMO_NOTIFY_MIN_SEVERITY")
    notify_timeout_seconds: float = 10.0

    @field_validator("ballast_symbol")
    @classmethod
    def _ballast_not_crypto(cls, v: str) -> str:
        if v and is_crypto_symbol(v):
            raise ValueError(
                f"BALLAST_SYMBOL {v!r} is a crypto pair; ballast must be an "
                "equity/ETF symbol (crypto vol makes it cash-substitute-unfit)"
            )
        return v

    @property
    def home_dir(self) -> Path:
        """Anchor for every relative path in this config.

        ``$DYNAMO_HOME`` when set; otherwise the repository root discovered by
        walking up from this module until a ``pyproject.toml`` appears; failing
        that (an installed wheel with no checkout), the CWD. Never CWD when a
        checkout is present, which is the whole point.
        """
        if self.dynamo_home:
            return Path(self.dynamo_home).expanduser().resolve()
        return _repo_root()

    def anchored(self, path: str | Path) -> Path:
        """Resolve ``path`` against :attr:`home_dir` unless it is absolute."""
        p = Path(path).expanduser()
        return p.resolve() if p.is_absolute() else (self.home_dir / p).resolve()

    @property
    def kill_switch_file(self) -> Path:
        """The ONE kill-switch file. Use this, never ``kill_switch_path``."""
        return self.anchored(self.kill_switch_path)

    @property
    def approvals_path(self) -> Path:
        return self.anchored(self.approvals_dir)

    @property
    def is_live(self) -> bool:
        return self.dynamo_env == DynamoEnv.LIVE

    @property
    def auto_approve(self) -> bool:
        # Live orders NEVER auto-approve regardless of config (doc 09 section 2).
        return self.execution_auto_approve and not self.is_live


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
