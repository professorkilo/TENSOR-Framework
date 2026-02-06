# Core Math Assurance (0.20260206d)

Generated: `2026-02-06T15:02:32Z`

## Scorecard
| Metric | Value |
| --- | --- |
| Overall score | 82.05 |
| Coverage score | 100.00 |
| Routing score | 100.00 |
| Information score | 24.25 |
| Robustness score | 86.03 |

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
| expectedStepsFromEntriesMean | 4.497928 |
| expectedStepsFromEntriesP90 | 4.497942 |
| absorptionDomainEntropy | 2.771825 |

### Information Theory
| Metric | Value |
| --- | --- |
| meanInformationGain | 0.658046 |
| negativeInformationGainNodeCount | 0 |

### Flow / Cut Theory
| Metric | Value |
| --- | --- |
| entryToTerminalEdgeConnectivity | 42 |
| entryToTerminalMinCutEdgeCount | 42 |

### Centrality Theory
| Metric | Value |
| --- | --- |
| betweennessCentralization | 0.01527 |
| betweennessGini | 0.759607 |

### Robustness / Percolation
| Metric | Value |
| --- | --- |
| targetedReachabilityAt5Percent | 0.860254 |

### Multi-objective Optimization
| Metric | Value |
| --- | --- |
| overallScore | 82.054593 |

### Formal Methods Checks
| Metric | Value |
| --- | --- |
| AG_branchTotality | True |
| AG_acyclic | True |
| AG_entryEventuallyConclude | True |

### Bayesian / Causal Readiness
| Metric | Value |
| --- | --- |
| fullArchetypeChainDomainCount | 7 |
| triangulationNodeRatio | 1 |

### Directed Topology Proxies
| Metric | Value |
| --- | --- |
| mergeNodeCount | 351 |
| diamondMotifCount | 40 |
