# Phase 1 Ingestion Log

**Purpose:** Track external data source ingestion (raw fetch only, no transformation)

**Phase 1 boundary:** Fetch → Save raw → Inspect → STOP. No cleaning, no merging, no database writes.

---

## World Bank Women, Business & Law (WBL)

**Date started:** 2026-01-26

**Purpose:** Legal equality framework to contextualize wage gap outcomes

**Source contract:** See `docs/phase1_source_wbl.md`

**Steps:**
1. Fetch WBL data via official API/download
2. Save raw to `data/raw/wbl/wbl_raw_YYYY-MM-DD.xlsx`
3. Log download metadata to `data/raw/wbl/download_log.jsonl`
4. Validate: check year range, duplicate keys, missingness
5. Document inspection notes in `docs/phase1_source_wbl.md`

**Commands:**

```bash
# Host machine
python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --out data/raw/wbl/wbl.xlsx

# Inside Docker container
docker exec -it gender-wage-gap-analysis-app-1 python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024

# Validate
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx
```

**Status:** 🚧 In progress

**Next steps:**
- [ ] Implement fetch script
- [ ] Test fetch on host
- [ ] Validate downloaded data
- [ ] Fill inspection notes in source contract
- [ ] Archive raw file with date stamp

---

## Future Sources (Phase 1)

### EIGE Gender Equality Index
- TBD

### OECD Gender Data
- TBD

### IPU Women in Parliament
- TBD

---

## Phase 2 (Future)

After Phase 1 completes for all sources:
- Design integration schema
- Implement cleaning/harmonization
- Create merged dataset
- Update Streamlit app
