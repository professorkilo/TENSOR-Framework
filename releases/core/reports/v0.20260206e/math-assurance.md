# Core Math Assurance (0.20260206e)

Generated: `2026-02-06T15:54:29Z`

## Scorecard
| Metric | Value |
| --- | --- |
| Overall score | 82.46 |
| Coverage score | 100.00 |
| Routing score | 100.00 |
| Information score | 24.12 |
| Robustness score | 88.19 |

## Publish Gates
Overall: **PASS**

| Gate | Metric | Rule | Value | Result |
| --- | --- | --- | --- | --- |
| dag_pass | graph_theory.dagPass | == True | True | pass |
| full_reachability | graph_theory.reachableRatio | >= 1.0 | 1 | pass |
| branch_totality | graph_theory.branchCompletenessRatio | >= 1.0 | 1 | pass |
| cross_domain_pivots | quality.crossDomainRatio | >= 0.25 | 0.648005 | pass |
| coverage_score | optimization.coverageScore | >= 100.0 | 100 | pass |
| routing_score | optimization.routingScore | >= 100.0 | 100 | pass |
| robustness_score | optimization.robustnessScore | >= 85.0 | 88.1932 | pass |
| targeted_resilience | robustness.targetedReachabilityAt5Percent | >= 0.85 | 0.881932 | pass |
| routing_centralization | centrality.betweennessCentralization | <= 0.02 | 0.016585 | pass |
| information_gain | information_theory.negativeInformationGainNodeCount | <= 0 | 0 | pass |
| overall_score | optimization.overallScore | >= 80.0 | 82.462334 | pass |

## Recommendations
- Current optimization profile is balanced; continue trend monitoring across releases.

## Theory Coverage
| Theory | Purpose |
| --- | --- |
| Graph Theory | Structural soundness: DAG, reachability, deterministic decision routing. |
| Matrix Theory | Coverage completeness and balance across Domain x Archetype. |
| Set Cover / Hitting Set | Minimal entry-path basis required to cover investigative feature space. |
| Markov Chain | Expected routing length and absorbing-outcome distribution under branch probabilities. |
| Information Theory | Uncertainty reduction and information gain across investigation decisions. |
| Flow / Cut Theory | Bottleneck and fault-tolerance identification via connectivity cuts. |
| Centrality Theory | Routing concentration and over-dependence detection. |
| Robustness / Percolation | Resilience under random and adversarial node removals. |
| Multi-objective Optimization | Unified scorecard for coverage, routing, robustness, and information quality. |
| Formal Methods Checks | Machine-checkable invariants for safety and liveness-style properties. |
| Bayesian / Causal Readiness | Readiness proxies for probabilistic confidence and causal interpretation overlays. |
| Directed Topology Proxies | Higher-order path-complex and merge-pattern shape indicators. |

## Key Metrics
### Graph Theory
| Metric | Value |
| --- | --- |
| branchCompletenessRatio | 1 |
| reachableRatio | 1 |
| rootCount | 1 |
| dagPass | True |

### Matrix Theory
| Metric | Value |
| --- | --- |
| matrixRank | 7 |
| nonZeroCellRatio | 1 |

### Set Cover / Hitting Set
| Metric | Value |
| --- | --- |
| configuredEntryCount | 16 |
| minimalEntrySetSizeByGreedy | 1 |
| entryCoverageRatio | 1 |

### Markov Chain
| Metric | Value |
| --- | --- |
| expectedStepsFromEntriesMean | 4.605688 |
| expectedStepsFromEntriesP90 | 4.52426 |
| absorptionDomainEntropy | 2.792028 |

### Information Theory
| Metric | Value |
| --- | --- |
| meanInformationGain | 0.659153 |
| negativeInformationGainNodeCount | 0 |

### Flow / Cut Theory
| Metric | Value |
| --- | --- |
| entryToTerminalEdgeConnectivity | 42 |
| entryToTerminalMinCutEdgeCount | 42 |

### Centrality Theory
| Metric | Value |
| --- | --- |
| betweennessCentralization | 0.016585 |
| betweennessGini | 0.74409 |

### Robustness / Percolation
| Metric | Value |
| --- | --- |
| targetedReachabilityAt5Percent | 0.881932 |

### Multi-objective Optimization
| Metric | Value |
| --- | --- |
| overallScore | 82.462334 |

### Formal Methods Checks
| Metric | Value |
| --- | --- |
| AG_branchTotality | True |
| AG_acyclic | True |
| AG_entryEventuallyTerminal | True |

### Bayesian / Causal Readiness
| Metric | Value |
| --- | --- |
| fullArchetypeChainDomainCount | 7 |
| triangulationNodeRatio | 0.996139 |

### Directed Topology Proxies
| Metric | Value |
| --- | --- |
| mergeNodeCount | 367 |
| diamondMotifCount | 47 |
