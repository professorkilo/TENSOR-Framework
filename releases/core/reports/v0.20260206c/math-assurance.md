# Core Math Assurance (0.20260206c)

Generated: `2026-02-06T14:23:29Z`

## Scorecard
| Metric | Value |
| --- | --- |
| Overall score | 76.38 |
| Coverage score | 67.86 |
| Routing score | 100.00 |
| Information score | 35.17 |
| Robustness score | 94.94 |

## Recommendations
- Add nodes for missing Domain x Archetype cells to maintain complete investigative coverage.

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
| matrixRank | 6 |
| nonZeroCellRatio | 0.678571 |

### Set Cover / Hitting Set
| Metric | Value |
| --- | --- |
| configuredEntryCount | 14 |
| minimalEntrySetSizeByGreedy | 1 |
| entryCoverageRatio | 1 |

### Markov Chain
| Metric | Value |
| --- | --- |
| expectedStepsFromEntriesMean | 2.488479 |
| expectedStepsFromEntriesP90 | 4.3125 |
| absorptionDomainEntropy | 1.66515 |

### Information Theory
| Metric | Value |
| --- | --- |
| meanInformationGain | 0.271179 |
| negativeInformationGainNodeCount | 0 |

### Flow / Cut Theory
| Metric | Value |
| --- | --- |
| entryToTerminalEdgeConnectivity | 3050 |
| entryToTerminalMinCutEdgeCount | 32 |

### Centrality Theory
| Metric | Value |
| --- | --- |
| betweennessCentralization | 0.004646 |
| betweennessGini | 0.480332 |

### Robustness / Percolation
| Metric | Value |
| --- | --- |
| targetedReachabilityAt5Percent | 0.949393 |

### Multi-objective Optimization
| Metric | Value |
| --- | --- |
| overallScore | 76.379419 |

### Formal Methods Checks
| Metric | Value |
| --- | --- |
| AG_branchTotality | True |
| AG_acyclic | True |
| AG_entryEventuallyConclude | False |

### Bayesian / Causal Readiness
| Metric | Value |
| --- | --- |
| fullArchetypeChainDomainCount | 4 |
| triangulationNodeRatio | 1 |

### Directed Topology Proxies
| Metric | Value |
| --- | --- |
| mergeNodeCount | 335 |
| diamondMotifCount | 543 |
