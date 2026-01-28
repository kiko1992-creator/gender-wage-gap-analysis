# Phase 1 WBL Execution Guide

**Status:** 🚧 Scaffolding complete, awaiting API configuration
**Date:** 2026-01-26

---

## Created Files

### Documentation
- ✅ `docs/phase1_source_wbl.md` - Source contract template
- ✅ `PHASE1_LOG.md` - Phase 1 activity log
- ✅ `data/reference/wbl/wbl_data_dictionary.md` - Variable dictionary

### Code
- ✅ `scripts/data_collection/_shared/io.py` - Shared utilities (JSONL, SHA256, timestamps)
- ✅ `scripts/data_collection/wbl/fetch_wbl.py` - WBL fetch script (needs API config)
- ✅ `scripts/data_collection/wbl/validate_wbl.py` - Validation script

### Folders
- ✅ `data/raw/wbl/` - Raw data storage
- ✅ `data/reference/wbl/` - Reference documentation
- ✅ `scripts/data_collection/_shared/` - Shared utilities
- ✅ `scripts/data_collection/wbl/` - WBL-specific scripts

---

## Next Steps (REQUIRED BEFORE EXECUTION)

### Step 1: Research WBL API Endpoint

**Action needed:** Determine correct World Bank WBL data access method.

**Options:**
1. **World Bank Data API** (api.worldbank.org/v2/)
2. **WBL Dedicated Portal** (wbl.worldbank.org)
3. **Bulk Download** (data.worldbank.org)

**Research tasks:**
- [ ] Visit https://wbl.worldbank.org/
- [ ] Check if bulk download is available
- [ ] Identify indicator codes for 8 WBL pillars
- [ ] Determine if API key is required
- [ ] Find country code format (ISO2/ISO3/name)

**Paste findings into:** `scripts/data_collection/wbl/fetch_wbl.py` at line 43 (WBL_CONFIG section)

### Step 2: Configure WBL_CONFIG

**Edit:** `scripts/data_collection/wbl/fetch_wbl.py`

**Replace placeholders:**
```python
WBL_CONFIG = {
    'base_url': 'PASTE ACTUAL URL',
    'indicator_codes': {
        'pillar_mobility': 'PASTE INDICATOR CODE',
        'pillar_workplace': 'PASTE INDICATOR CODE',
        # ... (8 pillars total)
    },
    'bulk_download_url': 'PASTE IF APPLICABLE',
}
```

### Step 3: Implement Fetch Logic

**Edit:** `scripts/data_collection/wbl/fetch_wbl.py` at line 118 (fetch_wbl_data method)

**Uncomment and adapt example code:**
- If using API: Loop through countries + years
- If using bulk download: Download full dataset, filter to target countries/years
- Return DataFrame with columns: country_code, year, wbl_index, pillar_*

---

## Execution Commands

### Prerequisites

```bash
# Install dependencies (if not already installed)
pip install pandas requests openpyxl
```

### Option A: Host Machine

```bash
# 1. Fetch WBL data (canonical 2020-2024)
python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --out data/raw/wbl/wbl.xlsx

# 2. Validate downloaded data
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx

# 3. Inspect raw file
# Open data/raw/wbl/wbl.xlsx in Excel/LibreOffice

# 4. Fill inspection notes
# Edit docs/phase1_source_wbl.md section 9 with:
#   - Rows/columns found
#   - Year coverage
#   - Country identifiers
#   - Variable columns
```

### Option B: Inside Docker Container

```bash
# 1. Enter container
docker exec -it gender-wage-gap-analysis-app-1 bash

# 2. Fetch WBL data
python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024

# 3. Validate
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx

# 4. Exit container
exit

# 5. Inspect on host
# data/raw/wbl/wbl.xlsx is volume-mapped, accessible on host
```

### Alternative: Extended Year Range

```bash
# Fetch wider range (historical context)
python scripts/data_collection/wbl/fetch_wbl.py --years 2015-2024 --out data/raw/wbl/wbl_extended.xlsx

# Canonical analysis will filter to 2020-2024 later
```

### Force Refresh

```bash
# Re-download even if file exists
python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --refresh
```

---

## Expected Output

### After Successful Fetch

```
data/raw/wbl/
├── wbl.xlsx                    # Raw WBL data (XLSX format)
└── download_log.jsonl          # Fetch metadata (timestamp, SHA256, row count)
```

### download_log.jsonl Format

```json
{
  "timestamp": "2026-01-26T15:30:00Z",
  "source": "WBL",
  "fetch_method": "API",
  "url": "https://...",
  "output_file": "wbl.xlsx",
  "sha256": "a1b2c3...",
  "years_requested": "2020-2024",
  "row_count": 150,
  "column_count": 15,
  "columns": ["country_code", "year", "wbl_index", "pillar_mobility", ...],
  "license": "CC BY 4.0",
  "notes": "Phase 1: Raw fetch only, no cleaning"
}
```

---

## Validation Checks

The validation script checks:

1. ✅ File exists
2. ✅ Loads successfully (XLSX format)
3. ✅ Required columns present (country, year, index/pillars)
4. ✅ Year range includes 2020-2024
5. ✅ No duplicate country-year keys
6. ✅ No null values in key columns
7. ✅ Target country coverage (EU27 + Balkans)

**Pass criteria:** All checks pass (warnings allowed)

---

## Troubleshooting

### Issue: "NotImplementedError: WBL fetch logic needs implementation"

**Cause:** API configuration incomplete (expected on first run)

**Fix:** Complete Step 1-3 above (research endpoint, configure WBL_CONFIG, implement fetch logic)

### Issue: "Missing dependency: pandas/requests"

**Fix:**
```bash
pip install pandas requests openpyxl
```

### Issue: "File not found" during validation

**Check:**
```bash
ls data/raw/wbl/
# Should show wbl.xlsx
```

### Issue: Docker volume mapping not working

**Fix:**
```bash
# Check docker-compose.yml has volume mapping:
# - ./data:/app/data
docker-compose restart
```

---

## Phase 1 Completion Checklist

- [ ] Research WBL API endpoint
- [ ] Configure `WBL_CONFIG` in fetch script
- [ ] Implement `fetch_wbl_data()` method
- [ ] Run fetch script successfully
- [ ] Validate downloaded data (all checks pass)
- [ ] Inspect XLSX file manually
- [ ] Fill section 9 in `docs/phase1_source_wbl.md` (inspection notes)
- [ ] Archive raw file with date stamp: `wbl_raw_2026-01-26.xlsx`
- [ ] Update `data/reference/wbl/wbl_data_dictionary.md` with actual columns
- [ ] Mark complete in `PHASE1_LOG.md`

**Phase 1 ends here.** No cleaning, no merging, no database writes, no Streamlit changes.

---

## Phase 2 Preview (Future)

After Phase 1 completes for all sources (WBL, EIGE, OECD, IPU):

1. Design integration schema (common country-year structure)
2. Implement cleaning/harmonization scripts
3. Create merged dataset combining wage gap + WBL + other sources
4. Add to PostgreSQL database
5. Update Streamlit dashboard with new variables
6. Run enhanced analyses (WBL as predictor of wage gap)

**For now:** Focus on Phase 1 only (raw fetch + validation).

---

## Support

**Questions about:**
- **WBL data source:** See `docs/phase1_source_wbl.md`
- **Fetch script:** See `scripts/data_collection/wbl/fetch_wbl.py` comments
- **Validation:** See `scripts/data_collection/wbl/validate_wbl.py`
- **Data dictionary:** See `data/reference/wbl/wbl_data_dictionary.md`
- **Overall Phase 1 plan:** See `PHASE1_LOG.md`
