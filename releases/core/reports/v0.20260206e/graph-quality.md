# Core Graph Quality Report (0.20260206e)

Generated: `2026-02-06T15:54:28Z`
Status: **PASS**

## Summary
| Metric | Value |
| --- | --- |
| Nodes | 588 |
| Edges | 1554 |
| Non-terminal nodes | 518 |
| Terminal nodes | 70 |
| Entry nodes | 16 |
| Reachable nodes | 588 |
| Cross-domain edges | 1007 |
| Cross-domain ratio | 0.6480 |
| Max fan-in | 6 |

## Domain Counts
| Domain | Count |
| --- | --- |
| Application | 75 |
| Cloud | 85 |
| Email | 80 |
| File | 73 |
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
| impact | 68 |
| terminal | 70 |

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
| Yes-edge semantic progression violations | 0 |
| Early-stage cross-domain yes violations | 0 |
