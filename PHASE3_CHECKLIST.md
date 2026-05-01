# Phase 3 (Data Integration & Analysis) — Conceptual Checklist

## Merge Prerequisites

1. All Phase 1 raw ingestion tasks (WBL, EIGE, OECD, IPU) must have passed their respective file-level validators with zero critical errors before any source enters the merge pipeline.
2. Each raw source must have a completed source-contract document specifying schema, temporal coverage, country coverage, and known missingness, so that merge logic can be defined deterministically.
3. A common merge key specification (country ISO-3 code + calendar year) must be agreed upon and every source must be confirmed mappable to that key without ambiguity.

## Data Quality Gates

4. After each pairwise merge, row counts must be reconciled against the Cartesian expectation (countries × years); any unmatched rows must be logged and classified as "expected gap" or "data error" before proceeding.
5. No variable may carry more than 30 percent missingness across the analytic sample; variables exceeding this threshold must be dropped or the source renegotiated rather than imputed.
6. Duplicate country-year observations must be resolved to exactly one record per key, with the resolution rule (e.g., prefer OFFICIAL reliability tier) documented before execution.
7. All continuous indicators must be checked for plausible ranges against their source-contract bounds; any out-of-range value must be flagged and traced back to its raw file before it propagates into models.

## Allowed Analyses

8. Descriptive statistics (means, medians, distributions) stratified by region (Balkans vs. EU) and by time period are the primary deliverable and should be completed before any modeling begins.
9. Regression models that estimate associations between legal-equality indices (WBL score, EIGE index) and observed wage gaps are permitted, with the explicit framing that coefficients represent conditional correlations, not causal effects.
10. Oaxaca-Blinder decomposition may be updated with the newly integrated covariates, provided the decomposition is reported with its full explained/unexplained partition and not reduced to a single headline number.

## Forbidden Analyses

11. No causal language (e.g., "X causes Y," "the effect of X on Y") is permitted anywhere in outputs; the observational, cross-country design cannot support causal identification and no instrumental-variable or quasi-experimental strategy is present.
12. Forecasting or extrapolation beyond the last observed data year must not be presented as prediction; any such output must be clearly labeled as a mechanical projection under stated assumptions and excluded from the main findings.
13. Subgroup analyses on fewer than three countries or fewer than five country-year observations are prohibited, as estimates would be unstable and misleading.

## Interpretation Guardrails

14. Every reported difference (e.g., Balkans mean gap vs. EU mean gap) must be accompanied by a measure of uncertainty or variability (confidence interval, standard error, or interquartile range); bare point estimates are not acceptable.
15. Discussion of results must acknowledge the key confounders that are unmeasured in this dataset -- notably sector composition, part-time work prevalence, and selection into formal employment -- and must state that residual gaps cannot be attributed solely to discrimination.
