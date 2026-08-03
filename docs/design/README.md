# Dynamo — Technical Design Document

Read in order. Each file carries a status header: `draft` → `reviewed` → `final`.

| # | Section | Owns |
|---|---|---|
| [00](00-goals-and-non-goals.md) | Goals & non-goals | Objective function, success metrics, live-readiness gate, budget contract |
| [01](01-architecture.md) | System architecture | Module map, dependency rule, topology, environments, cross-cutting concerns |
| [02](02-data-platform.md) | Data platform & universe | Providers, schemas, three-tier universe, quality gates, cadences |
| [03](03-claude-committee.md) | Claude committee v2 | Agent structure, schemas, caching/batches, disagreement, cost kill-switch |
| [04](04-quant-models.md) | Quant models | Features, labels, walk-forward training, calibration, model registry |
| [05](05-strategies-and-scoring.md) | Strategies & scoring | Quality score, tactical strategies, sleeve design, ranking |
| [06](06-risk-engine.md) | **Risk engine** | The veto contract, limits, drawdown ladder, sizing, VaR/stress, regime |
| [07](07-portfolio-construction.md) | Portfolio construction | Rule-based construction, cash policy, rebalancing, attribution tags |
| [08](08-backtesting-and-validation.md) | Backtesting & validation | Two engines, overfitting defenses, **the promotion pipeline** |
| [09](09-execution.md) | Execution | Order lifecycle, reconciliation, live gate, kill-switch, DecisionContext capture |
| [10](10-learning-loop.md) | Learning loop | Post-trade reviews, failure taxonomy, calibration, journal, drift |
| [11](11-dashboard-and-api.md) | Dashboard & API | API surface, screen inventory, two-phase frontend |
| [12](12-operations.md) | Operations | CI/CD, backups, monitoring, alert catalog, runbooks |
| [13](13-risk-register.md) | Risk register | Project risks, mitigations, tripwires |
| [14](14-design-review.md) | Design review | Principal-engineer self-review findings & resolutions |

Decisions with alternatives live in [`../adr/`](../adr/).
