# Data Dictionary: Gender Wage Gap Analysis
## Final Analytical Dataset Documentation

**Last Updated:** 2026-02-06
**Dataset Period:** 2009-2024 (varies by country and indicator)
**Geographic Coverage:** 12 countries (3 Balkans, 9 EU)
**Purpose:** Reproducible empirical analysis of gender wage disparities

---

## 1. OUTCOME VARIABLE

### 1.1 Gender Wage Gap Metric

| Variable Name | `wage_gap_pct` |
|---------------|----------------|
| **Definition** | Unadjusted gender pay gap in gross hourly earnings, measured as the percentage difference between average male and female earnings |
| **Formula** | `((Male_avg_wage - Female_avg_wage) / Male_avg_wage) × 100` |
| **Unit / Scale** | Percentage (0-100). Positive values indicate men earn more than women; negative values indicate women earn more than men |
| **Source Dataset** | Multiple sources consolidated: Eurostat (Structure of Earnings Survey, indicator SDG_05_20), National Statistical Offices (North Macedonia, Serbia, Montenegro), ILO statistics |
| **Source Institution** | Eurostat (EU countries), State Statistical Office of North Macedonia, Statistical Office of the Republic of Serbia, Statistical Office of Montenegro, International Labour Organization |
| **Temporal Coverage** | 2009-2024 (varies by country: North Macedonia 2009-2023, Serbia 2009-2024, EU countries primarily 2021-2023) |
| **Known Limitations** | <ul><li>**Measurement heterogeneity**: Different national statistical methodologies for wage surveys; Eurostat harmonizes EU data but Balkan countries may use different definitions</li><li>**Sectoral coverage**: Some countries exclude micro-enterprises (<10 employees), agriculture, or public administration</li><li>**Selection bias**: Excludes unpaid work, informal economy workers, and self-employed individuals</li><li>**Timing**: Survey reference periods differ (some annual averages, some October snapshots)</li><li>**Unadjusted**: Does not control for occupation, experience, education, hours worked, or other wage-determining factors; differences may reflect workforce composition rather than discrimination alone</li><li>**Comparability caveat**: Direct cross-country comparisons should account for institutional differences in data collection</li></ul> |

### 1.2 Average Monthly Wage

| Variable Name | `avg_monthly_wage` |
|---------------|-------------------|
| **Definition** | Mean gross monthly earnings in local currency, disaggregated by gender |
| **Unit / Scale** | Local currency units (not standardized across countries; nominal values) |
| **Source Dataset** | Same as wage_gap_pct (Eurostat SES, national labor force surveys) |
| **Source Institution** | National statistical offices, Eurostat |
| **Known Limitations** | <ul><li>**Nominal values**: Not adjusted for inflation or purchasing power parity; temporal comparisons require deflation</li><li>**Currency differences**: Cross-country comparisons invalid without PPP adjustment</li><li>**Full-time equivalent bias**: May not distinguish full-time vs. part-time workers consistently</li></ul> |

---

## 2. LEGAL & INSTITUTIONAL INDICATORS

### 2.1 Women, Business and the Law (WBL) Index

| Variable Name | `WBL_Index_Overall` |
|---------------|---------------------|
| **Definition** | Composite index measuring legal equality between women and men across eight dimensions of economic participation over the life cycle |
| **Unit / Scale** | Score from 0 to 100 (100 = full legal equality; 0 = no legal equality). Calculated as unweighted average of eight pillar scores |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.INDX` |
| **Temporal Coverage** | 2020-2024 (annual updates) |
| **Known Limitations** | <ul><li>**De jure vs. de facto**: Measures laws on the books, not actual enforcement or lived experience; high scores may coexist with weak implementation</li><li>**Binary coding**: Most indicators are yes/no questions; does not capture nuance in legal provisions or quality of protections</li><li>**Urban bias**: Primarily reflects legal frameworks in largest business cities; rural areas may differ</li><li>**Scope exclusion**: Does not measure cultural norms, social acceptance, access to justice, or informal sector conditions</li><li>**Aggregation assumptions**: Equal weighting of pillars assumes all dimensions equally important; no sensitivity analysis provided</li><li>**Measurement discontinuity**: Methodology has been revised over time; historical comparisons require caution</li></ul> |

### 2.2 WBL Pillar: Mobility

| Variable Name | `WBL_Mobility` |
|---------------|----------------|
| **Definition** | Legal constraints on women's freedom of movement (e.g., ability to travel outside the home, choose where to live, obtain passport, travel abroad like men) |
| **Unit / Scale** | Score 0-100 (100 = no legal mobility restrictions) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.MO` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: Most EU and Balkan countries score near 100; limited variation may reduce analytical utility |

### 2.3 WBL Pillar: Workplace

| Variable Name | `WBL_Workplace` |
|---------------|-----------------|
| **Definition** | Gender-based job restrictions (e.g., laws prohibiting women from working in certain industries, night work, or hazardous jobs when men face no such restrictions) |
| **Unit / Scale** | Score 0-100 (100 = no gender-based workplace restrictions) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.WK` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: Some protective legislation (e.g., restrictions on hazardous work during pregnancy) may be classified as restrictions, though intended as protections |

### 2.4 WBL Pillar: Pay

| Variable Name | `WBL_Pay` |
|---------------|-----------|
| **Definition** | Existence of laws mandating equal remuneration for work of equal value, and whether gender non-discrimination in employment is mandated |
| **Unit / Scale** | Score 0-100 (100 = equal pay laws fully in place) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.PA` |
| **Known Limitations** | Same as WBL_Index_Overall; **critical**: High scores indicate law exists, not that pay equity is achieved; enforcement and compliance data not captured |

### 2.5 WBL Pillar: Marriage

| Variable Name | `WBL_Marriage` |
|---------------|----------------|
| **Definition** | Legal equality within marriage (e.g., equal inheritance rights, equal authority over children, ability to obtain divorce on same grounds, remarriage rights) |
| **Unit / Scale** | Score 0-100 (100 = full equality in marriage laws) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.MA` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: Religious or customary law may operate alongside civil law, not fully captured |

### 2.6 WBL Pillar: Parenthood

| Variable Name | `WBL_Parenthood` |
|---------------|-----------------|
| **Definition** | Parental leave and childcare provisions (e.g., paid maternity leave duration, paternity leave availability, parental leave, dismissal protections for pregnant workers) |
| **Unit / Scale** | Score 0-100 (100 = comprehensive parental leave protections) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.PR` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: <ul><li>Does not measure generosity of benefits (e.g., 50% wage replacement vs. 100%)</li><li>Does not capture employer compliance or take-up rates</li><li>Does not measure availability or affordability of childcare services</li></ul> |

### 2.7 WBL Pillar: Entrepreneurship

| Variable Name | `WBL_Entrepreneurship` |
|---------------|------------------------|
| **Definition** | Legal constraints on women's entrepreneurship (e.g., ability to register business, open bank account, sign contracts, access credit without spousal authorization) |
| **Unit / Scale** | Score 0-100 (100 = no legal barriers to entrepreneurship) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.EN` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: Does not measure access to finance in practice (credit availability, collateral requirements, informal lending barriers) |

### 2.8 WBL Pillar: Assets

| Variable Name | `WBL_Assets` |
|---------------|--------------|
| **Definition** | Gender equality in property and inheritance rights (e.g., sons and daughters have equal inheritance rights, spouses have equal rights to property during marriage and after divorce) |
| **Unit / Scale** | Score 0-100 (100 = equal property rights) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.AS` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: Customary and religious law may govern inheritance in practice, diverging from civil law |

### 2.9 WBL Pillar: Pension

| Variable Name | `WBL_Pension` |
|---------------|---------------|
| **Definition** | Gender equality in pension and retirement provisions (e.g., mandatory retirement age, pension calculation, survivor benefits) |
| **Unit / Scale** | Score 0-100 (100 = gender parity in pension laws) |
| **Source Dataset** | World Bank Women, Business and the Law Database |
| **Source Institution** | World Bank Group |
| **API Indicator Code** | `SG.LAW.NODC.PE` |
| **Known Limitations** | Same as WBL_Index_Overall; additionally: <ul><li>Does not capture adequacy of pension benefits</li><li>Does not measure gender pension gap arising from career interruptions or wage gaps</li><li>Lower mandatory retirement age for women (common in some countries) may be coded as inequality even if women prefer earlier retirement</li></ul> |

---

## 3. GEOGRAPHIC & TEMPORAL IDENTIFIERS

### 3.1 Country Code

| Variable Name | `country_code` |
|---------------|----------------|
| **Definition** | Two-letter country identifier following international standard |
| **Unit / Scale** | ISO 3166-1 alpha-2 codes (e.g., MK = North Macedonia, RS = Serbia, BG = Bulgaria) |
| **Source Dataset** | Not applicable (standardization variable) |
| **Source Institution** | International Organization for Standardization (ISO) |
| **Known Limitations** | <ul><li>**Kosovo designation**: Uses 'XK' (user-assigned code, not official ISO); politically contested</li><li>**Historical changes**: North Macedonia formerly coded as FYR Macedonia in some sources; code 'MK' standardized post-2019 Prespa Agreement</li></ul> |

### 3.2 Country Name

| Variable Name | `country_name` |
|---------------|----------------|
| **Definition** | Full official name of country in English |
| **Unit / Scale** | Text string |
| **Source Dataset** | Not applicable (descriptor variable) |
| **Source Institution** | United Nations / ISO standard names |
| **Known Limitations** | Name changes (e.g., "Macedonia" vs. "North Macedonia") may create inconsistencies in historical data |

### 3.3 Year

| Variable Name | `year` |
|---------------|--------|
| **Definition** | Calendar year of observation or survey reference period |
| **Unit / Scale** | Four-digit integer (2009-2024) |
| **Source Dataset** | Derived from underlying data sources |
| **Source Institution** | Not applicable (temporal identifier) |
| **Known Limitations** | <ul><li>**Survey timing**: Some wage surveys conducted in Q4 of year t but released in year t+1; coding follows survey year, not publication year</li><li>**Missing years**: Not all countries have data for all years; irregular data availability, especially for Balkan countries</li><li>**Structural breaks**: COVID-19 pandemic (2020-2021) and policy changes may create non-comparability across years</li></ul> |

### 3.4 Region

| Variable Name | `region` |
|---------------|----------|
| **Definition** | Geographic-political classification |
| **Unit / Scale** | Categorical: "EU" (European Union member states) or "Balkans" (Western Balkan non-EU countries: Albania, Bosnia and Herzegovina, Kosovo, North Macedonia, Montenegro, Serbia) |
| **Source Dataset** | Not applicable (analytical classification) |
| **Source Institution** | Not applicable (author-defined) |
| **Known Limitations** | <ul><li>**Classification ambiguity**: Croatia, Bulgaria, Slovenia are both "EU" and geographically Balkan; coded as "EU" based on institutional membership</li><li>**Temporal validity**: Classification reflects 2020-2024 status; historical analyses should note EU enlargement (Croatia joined 2013, Bulgaria/Romania 2007)</li><li>**Heterogeneity within categories**: "Balkans" groups countries with diverse political and economic conditions; "EU" includes both Western and Eastern European states with significant institutional differences</li><li>**No causal interpretation**: Regional classification is descriptive, not a treatment variable</li></ul> |

---

## 4. ECONOMIC & LABOR MARKET CONTROLS

### 4.1 GDP per Capita

| Variable Name | `gdp_per_capita` |
|---------------|------------------|
| **Definition** | Gross Domestic Product per capita, current prices |
| **Unit / Scale** | U.S. dollars, current (nominal) prices; not PPP-adjusted in base dataset |
| **Source Dataset** | World Bank World Development Indicators (WDI) or Eurostat National Accounts |
| **Source Institution** | World Bank, Eurostat |
| **Temporal Coverage** | Annual, 2009-2024 |
| **Known Limitations** | <ul><li>**Nominal values**: Not adjusted for inflation; real GDP per capita (constant prices or PPP-adjusted) preferred for temporal and cross-country comparisons</li><li>**Distribution ignorance**: Mean value; does not reflect inequality or median income</li><li>**Informal economy**: Underestimates economic activity in countries with large informal sectors (more significant in Balkans)</li><li>**Valuation differences**: Exchange rate fluctuations affect comparability</li></ul> |

### 4.2 Female Labor Force Participation Rate

| Variable Name | `female_lfp` |
|---------------|--------------|
| **Definition** | Percentage of female working-age population (typically ages 15-64) that is economically active (employed or actively seeking employment) |
| **Unit / Scale** | Percentage (0-100) |
| **Source Dataset** | ILO Statistics (ILOSTAT), Eurostat Labour Force Survey (LFS) |
| **Source Institution** | International Labour Organization, Eurostat |
| **Temporal Coverage** | Annual, 2009-2024 |
| **Known Limitations** | <ul><li>**Age definition**: Working-age population definition varies (15-64 vs. 15+); confirm source definition</li><li>**Discouraged workers**: Excludes individuals who have stopped job searching; may underestimate labor supply</li><li>**Informal work**: May not fully capture informal employment, especially in agriculture and domestic work</li><li>**Part-time/full-time**: Does not distinguish part-time vs. full-time participation; FTE adjustments not made</li><li>**Seasonal variation**: Annual averages may mask seasonal employment patterns</li></ul> |

### 4.3 Male Labor Force Participation Rate

| Variable Name | `male_lfp` |
|---------------|------------|
| **Definition** | Percentage of male working-age population (typically ages 15-64) that is economically active (employed or actively seeking employment) |
| **Unit / Scale** | Percentage (0-100) |
| **Source Dataset** | ILO Statistics (ILOSTAT), Eurostat Labour Force Survey (LFS) |
| **Source Institution** | International Labour Organization, Eurostat |
| **Temporal Coverage** | Annual, 2009-2024 |
| **Known Limitations** | Same as female_lfp |

### 4.4 Labor Force Participation Gap

| Variable Name | `lfp_gap` |
|---------------|-----------|
| **Definition** | Gender gap in labor force participation, calculated as male LFP minus female LFP |
| **Formula** | `male_lfp - female_lfp` |
| **Unit / Scale** | Percentage points. Positive values indicate higher male participation; zero indicates parity |
| **Source Dataset** | Derived variable from `female_lfp` and `male_lfp` |
| **Source Institution** | Not applicable (calculated) |
| **Known Limitations** | <ul><li>**Absolute gap metric**: Does not account for baseline participation levels; ratio measure (female/male) may be more appropriate in some contexts</li><li>**Compositional neutrality**: Does not indicate whether gap narrowing is due to female increases or male decreases</li></ul> |

### 4.5 Unemployment Rate

| Variable Name | `unemployment` |
|---------------|----------------|
| **Definition** | Total unemployment rate (all genders), percentage of labor force that is unemployed and actively seeking work |
| **Unit / Scale** | Percentage (0-100) |
| **Source Dataset** | ILO Statistics (ILOSTAT), Eurostat Labour Force Survey (LFS) |
| **Source Institution** | International Labour Organization, Eurostat |
| **Temporal Coverage** | Annual, 2009-2024 |
| **Known Limitations** | <ul><li>**ILO definition**: Requires active job search in recent period (usually 4 weeks); excludes discouraged workers</li><li>**Underemployment**: Does not capture involuntary part-time work or underemployment</li><li>**Youth unemployment**: Aggregate rate may mask high youth unemployment in some countries</li><li>**Survey vs. registered**: Eurostat uses survey-based measure; some national statistics use registered unemployment (lower figures)</li></ul> |

---

## 5. DATA QUALITY & RELIABILITY CLASSIFICATION

### 5.1 Reliability Indicator

| Variable Name | `reliability` |
|---------------|---------------|
| **Definition** | Data quality classification indicating trustworthiness of wage gap estimate |
| **Unit / Scale** | Categorical: <ul><li>**OFFICIAL**: Data from national statistical offices or Eurostat; methodologically standardized</li><li>**RESEARCH**: Peer-reviewed academic publications or reports from international organizations (ILO, UNECE)</li><li>**ESTIMATE**: Model-based projections, synthetic estimates, or author calculations using indirect methods</li></ul> |
| **Source Dataset** | Not applicable (metadata variable assigned during data validation) |
| **Source Institution** | Not applicable (author-assigned) |
| **Known Limitations** | <ul><li>**Subjective classification**: Boundaries between categories not always clear; some "RESEARCH" sources may be more rigorous than "OFFICIAL" data from countries with weaker statistical capacity</li><li>**Heterogeneity within categories**: "OFFICIAL" data from Eurostat vs. national offices may differ in quality</li><li>**Recommendation**: Sensitivity analyses should test robustness by excluding "ESTIMATE" category</li></ul> |

### 5.2 Data Source Indicator

| Variable Name | `data_source` |
|---------------|---------------|
| **Definition** | Specific institution or publication that provided the wage gap data |
| **Unit / Scale** | Text string (e.g., "Eurostat SDG_05_20", "State Statistical Office", "ILO Study") |
| **Source Dataset** | Not applicable (provenance metadata) |
| **Source Institution** | Not applicable (descriptor) |
| **Known Limitations** | <ul><li>**Traceability**: Not all original sources are publicly accessible; some reports may require institutional access</li><li>**Version control**: Does not specify data vintage or revision status; Eurostat and World Bank occasionally revise historical data</li></ul> |

---

## 6. MISSING DATA DOCUMENTATION

### 6.1 Pattern of Missingness

| Aspect | Description |
|--------|-------------|
| **Countries with gaps** | Albania, Bosnia and Herzegovina, Kosovo: Excluded from analytical dataset due to insufficient reliable wage data (>50% missing years) |
| **Time period gaps** | Balkan countries: Irregular data collection, gaps in 2010-2015 period. EU countries: Sparse data before 2020; Eurostat harmonization improves coverage post-2018 |
| **WBL indicators** | Minor missingness (5-10%) for recent years (2023-2024) where reports not yet finalized; Kosovo ('XK') not covered by World Bank WBL index |
| **Labor market indicators** | LFP and unemployment generally complete for all 12 countries in analytical dataset; GDP per capita complete |

### 6.2 Imputation Approach

| Variable | Imputation Method |
|----------|------------------|
| **wage_gap_pct** | **No imputation**. Missing years excluded from analysis; listwise deletion used |
| **WBL indicators** | **Carry-forward from previous year** (if missing year is 2023-2024 only). Justification: Legal frameworks change slowly; one-year lag unlikely to introduce bias. Sensitivity test: Exclude imputed observations |
| **GDP, LFP, unemployment** | Linear interpolation for single missing years within country time series (rare, <2% of cases). Extrapolation avoided |

**Missing data handling recommendation for analysis:**
- Use multiple imputation with chained equations (MICE) for robustness checks
- Report results with and without imputed observations
- Assess whether missingness is related to outcome variable (non-random missingness)

---

## 7. DATASET LINEAGE & VERSIONING

| Attribute | Value |
|-----------|-------|
| **Primary dataset files** | `data/processed/validated_wage_data.csv` (individual-level disaggregation), `data/processed/ml_features_clustered.csv` (country-year aggregate) |
| **Raw data locations** | `data/raw/` (original files), `data/reference/official_gpg_data.csv` (Eurostat benchmark) |
| **Data collection scripts** | `scripts/data_collection/fetch_worldbank_wbl.py` |
| **Validation scripts** | `scripts/comprehensive_data_pipeline.py` |
| **Last validation date** | 2024-01-21 (WBL data); wage data validated December 2025 |
| **Recommended citation** | [Author/Institution]. (2025). *Gender Wage Gap Analysis Dataset: EU and Balkan Countries, 2009-2024* [Data file and code book]. Retrieved from [repository URL] |

---

## 8. STATISTICAL CONSIDERATIONS & USAGE NOTES

### 8.1 Appropriate Uses
- Descriptive analysis of wage gap trends within and across countries
- Exploratory analysis of correlations between legal frameworks and wage outcomes
- Panel regression with country and year fixed effects (accounting for unobserved heterogeneity)
- Cluster analysis and comparative case studies

### 8.2 Methodological Cautions
- **Endogeneity**: WBL indicators and wage gaps may both be caused by unobserved factors (culture, institutions); regression coefficients should not be interpreted as causal effects without identification strategy (IV, DID, RDD)
- **Reverse causality**: Countries with high gender equality may both legislate protections and experience smaller wage gaps; directionality ambiguous
- **Omitted variables**: Dataset lacks controls for education distribution, occupational segregation, experience, industry structure—critical confounders
- **Small sample**: 12 countries, limited temporal variation; low statistical power for interaction terms or heterogeneity analysis
- **Clustered standard errors**: Required in regression analysis (cluster by country); failure to account for within-country correlation will underestimate standard errors

### 8.3 Recommended Extensions
- Add EIGE Gender Equality Index (cultural norms)
- Include childcare cost and availability (OECD Family Database)
- Merge occupational segregation indices (Eurostat LFS)
- Add political representation (IPU women in parliament data)

---

## 9. VARIABLE SUMMARY TABLE

| Variable Name | Type | Range/Categories | Source | Coverage |
|---------------|------|------------------|--------|----------|
| `wage_gap_pct` | Continuous | -10 to 30% | Eurostat, NSOs | 2009-2024, 12 countries |
| `avg_monthly_wage` | Continuous | 38,500 - 55,800 | Eurostat, NSOs | 2009-2024, 12 countries |
| `WBL_Index_Overall` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Mobility` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Workplace` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Pay` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Marriage` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Parenthood` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Entrepreneurship` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Assets` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `WBL_Pension` | Continuous | 0-100 | World Bank | 2020-2024, 12 countries |
| `country_code` | Categorical | ISO alpha-2 codes | ISO 3166-1 | All observations |
| `country_name` | Categorical | 12 country names | ISO | All observations |
| `year` | Discrete | 2009-2024 | NSOs, Eurostat | Varies by country |
| `region` | Categorical | "EU", "Balkans" | Author classification | All observations |
| `gdp_per_capita` | Continuous | 6,800 - 55,800 USD | World Bank WDI | 2009-2024 |
| `female_lfp` | Continuous | 0-100% | ILO, Eurostat | 2009-2024 |
| `male_lfp` | Continuous | 0-100% | ILO, Eurostat | 2009-2024 |
| `lfp_gap` | Continuous | -20 to +30 pp | Derived | 2009-2024 |
| `unemployment` | Continuous | 0-20% | ILO, Eurostat | 2009-2024 |
| `reliability` | Categorical | OFFICIAL, RESEARCH, ESTIMATE | Author-assigned | All observations |
| `data_source` | Categorical | Text labels | Metadata | All observations |

---

## 10. REFERENCES & FURTHER READING

### Data Sources
- **Eurostat**. (2024). *Structure of Earnings Survey: Gender Pay Gap (SDG_05_20)*. European Commission. https://ec.europa.eu/eurostat/web/labour-market/earnings/database
- **World Bank**. (2024). *Women, Business and the Law 2024*. Washington, DC: World Bank. https://wbl.worldbank.org/
- **ILO**. (2024). *ILOSTAT Database: Labour Force Statistics*. International Labour Organization. https://ilostat.ilo.org/
- **World Bank**. (2024). *World Development Indicators*. https://databank.worldbank.org/source/world-development-indicators

### Methodological References
- Blau, F. D., & Kahn, L. M. (2017). The gender wage gap: Extent, trends, and explanations. *Journal of Economic Literature*, 55(3), 789-865.
- European Commission. (2021). *2021 Report on Gender Equality in the EU*. Luxembourg: Publications Office of the European Union.
- Weichselbaumer, D., & Winter-Ebmer, R. (2005). A meta-analysis of the international gender wage gap. *Journal of Economic Surveys*, 19(3), 479-511.

---

## DOCUMENT VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-06 | Initial draft data dictionary created |

---

**Contact for Data Questions:**
See project repository README for contributor contact information.

**License:**
Data sources retain original licenses (World Bank: CC BY 4.0; Eurostat: [Eurostat Copyright Policy](https://ec.europa.eu/eurostat/about/policies/copyright)); analytical dataset compiled for academic research purposes.
