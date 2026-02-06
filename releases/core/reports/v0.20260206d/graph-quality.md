# Core Graph Quality Report (0.20260206d)

Generated: `2026-02-06T15:02:32Z`
Status: **PASS**

## Summary
| Metric | Value |
| --- | --- |
| Nodes | 580 |
| Edges | 1548 |
| Non-terminal nodes | 516 |
| Terminal nodes | 64 |
| Entry nodes | 16 |
| Reachable nodes | 580 |
| Cross-domain edges | 1008 |
| Cross-domain ratio | 0.6512 |
| Max fan-in | 6 |

## Domain Counts
| Domain | Count |
| --- | --- |
| Application | 75 |
| Cloud | 85 |
| Email | 80 |
| File | 65 |
| Host | 95 |
| Identity | 95 |
| Network | 85 |

## Archetype Counts
| Archetype | Count |
| --- | --- |
| detect | 80 |
| validate | 78 |
| classify | 76 |
| scope | 74 |
| correlate | 72 |
| attribute | 70 |
| impact | 66 |
| terminal | 64 |

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
