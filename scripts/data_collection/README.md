# Data Collection Scripts

This directory contains scripts to fetch external data for enriching the gender wage gap analysis.

## Setup

```bash
# Install requirements
cd scripts/data_collection
pip install -r requirements_datacollection.txt
```

## Available Scripts

### 1. `fetch_worldbank_wbl.py`
Fetches **World Bank Women, Business and the Law** data.

**What it provides:**
- Legal equality indicators across 8 dimensions
- Scores from 0-100 (100 = full legal equality)
- Coverage: All EU27 + Balkans countries
- Years: 2020-2024

**Usage:**
```bash
python fetch_worldbank_wbl.py
```

**Output:**
- `data/raw/world_bank/wbl_long_latest.csv` - Long format (one row per country-year-indicator)
- `data/raw/world_bank/wbl_wide_latest.csv` - Wide format (one row per country-year)
- Metadata JSON file with fetch details

**Indicators fetched:**
1. Overall WBL Index
2. Mobility (travel/work restrictions)
3. Workplace (gender-based job restrictions)
4. Pay (equal remuneration laws)
5. Marriage (equality in marriage)
6. Parenthood (parental leave, childcare)
7. Entrepreneurship (business rights)
8. Assets (property rights)
9. Pension (retirement equality)

---

## Planned Scripts (To Be Implemented)

### 2. `fetch_eige.py`
European Institute for Gender Equality Index
- Gender equality scores across 6 domains
- EU countries only
- **Status:** Planned

### 3. `fetch_eurostat_lfs.py`
Eurostat Labour Force Survey
- Employment structure by gender
- Part-time rates, occupational segregation
- **Status:** Planned

### 4. `fetch_oecd_family.py`
OECD Family Database
- Childcare policies and costs
- Parental leave duration and compensation
- **Status:** Planned

### 5. `fetch_ipu_political.py`
Inter-Parliamentary Union Database
- Women in parliament (% seats)
- Women in ministerial positions
- **Status:** Planned

---

## Data Flow

```
1. Raw Data Fetch (this directory)
   ↓
2. Data Cleaning (scripts/data_cleaning/)
   ↓
3. Database Loading (scripts/database_loading/)
   ↓
4. PostgreSQL Database
   ↓
5. Streamlit Dashboard
```

---

## File Naming Convention

**Raw files:**
- `{source}_{format}_{timestamp}.csv` - Versioned files
- `{source}_{format}_latest.csv` - Latest version (for convenience)

**Examples:**
- `wbl_long_20240121_143052.csv`
- `wbl_long_latest.csv`

---

## Error Handling

All scripts include:
- ✅ Timeout handling (10 seconds per request)
- ✅ Retry logic for network errors
- ✅ Missing data warnings
- ✅ Metadata tracking (fetch date, source URLs)

---

## Next Steps After Fetching Data

1. **Review the data:**
   ```bash
   cd data/raw/world_bank
   head wbl_wide_latest.csv
   ```

2. **Check for missing values:**
   - Open in Excel or
   - Use `pandas` to inspect

3. **Clean the data:**
   - Run cleaning scripts (to be created in `scripts/data_cleaning/`)

4. **Load to database:**
   - Run loading scripts (to be created in `scripts/database_loading/`)

5. **Integrate into analysis:**
   - Update Streamlit pages to use new indicators

---

## Troubleshooting

**"Connection timeout":**
- Check internet connection
- World Bank API might be temporarily down
- Try again later

**"No data fetched":**
- Check if country codes are correct
- Verify years are available in the API
- Review API documentation for changes

**"Missing values for some countries":**
- Normal - not all countries report all indicators
- Document missing data in metadata
- Consider imputation strategies

---

## Contributing

When adding new data fetcher scripts:

1. Follow the template structure in `fetch_worldbank_wbl.py`:
   - Clear docstrings
   - Error handling
   - Metadata saving
   - Both long and wide format outputs

2. Update this README with:
   - Script description
   - What data it provides
   - Usage instructions

3. Add any new dependencies to `requirements_datacollection.txt`

---

## Data Sources Documentation

Full documentation of all planned data sources: See `DATA_ENRICHMENT_PLAN.md` in project root.

---

**Last Updated:** 2024-01-21
