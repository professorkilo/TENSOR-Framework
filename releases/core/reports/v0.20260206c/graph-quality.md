# Core Graph Quality Report (0.20260206c)

Generated: `2026-02-06T13:01:26Z`
Status: **PASS**

## Summary
| Metric | Value |
| --- | --- |
| Nodes | 520 |
| Edges | 1005 |
| Non-terminal nodes | 335 |
| Terminal nodes | 185 |
| Entry nodes | 14 |
| Reachable nodes | 520 |
| Cross-domain edges | 657 |
| Cross-domain ratio | 0.6537 |
| Max fan-in | 3 |

## Domain Counts
| Domain | Count |
| --- | --- |
| Application | 50 |
| Cloud | 85 |
| Email | 65 |
| File | 35 |
| Host | 105 |
| Identity | 95 |
| Network | 85 |

## Archetype Counts
| Archetype | Count |
| --- | --- |
| detect | 44 |
| validate | 43 |
| classify | 42 |
| scope | 42 |
| correlate | 42 |
| attribute | 41 |
| impact | 133 |
| terminal | 133 |

## Entry Coverage By Domain
| Domain | Has Entry |
| --- | --- |
| Application | yes |
| Cloud | yes |
| Email | yes |
| File | yes |
| Host | yes |
| Identity | yes |
| Network | yes |

## Checks
| Check | Result |
| --- | --- |
| All nodes reachable from entries | pass |
| DAG | pass |
| Cross-domain ratio >= threshold | pass |
| Branch determinism violations | 0 |
