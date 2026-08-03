# ADR-0006 — Clean-Architecture-lite: ports + adapters + one enforced dependency rule

**Status:** accepted (user-approved) · 2026-08-03

## Context
The spec asks for Clean Architecture, SOLID, DI, and the repository pattern. Full ceremonial
Clean Architecture (use-case class per verb, interface per repository, DTO mapping layers)
exists to coordinate large teams; on a solo project it taxes velocity without adding safety.

## Decision
Keep: a pure `domain` package (entities + `Protocol` ports, imports nothing), core logic that
imports domain only, adapters implementing ports, repositories per aggregate *cluster*, and a
hand-rolled composition root (`wiring.py` with typed factories). Enforce exactly two
import-linter contracts in CI: domain purity and core-does-not-import-adapters.

Skip: DI frameworks, per-entity repository interfaces, DTO mappers, layered exception
hierarchies.

## Consequences
Swappable brokers/providers/models, testable core, honest boundaries — at solo speed. The
boundary that must stay pure is `domain` + `risk`; pyright strict enforces typing there.
Abstraction elsewhere is added only when a second implementation actually appears.
