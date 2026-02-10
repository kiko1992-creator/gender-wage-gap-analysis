# Phase 1 Source Contract — EIGE Gender Equality Index

## 0) Hard Stop (Phase 1 boundary)
Phase 1 ends after:
- raw EIGE data is fetched and saved in `data/raw/eige/`
- file-level validation report is written
- inspection note documents worksheets/structure (no cell parsing)

No cleaning, no merging, no database writes, no Streamlit changes.

## 1) Purpose in this project
Gender equality measurement across multiple domains to contextualize wage gap patterns in EU countries.

## 2) Official source description
**Source:** European Institute for Gender Equality (EIGE)
**Dataset:** Gender Equality Index
**URL:** https://eige.europa.eu/gender-equality-index
**Coverage:** EU27 countries
**Temporal:** Biennial updates (2013, 2015, 2017, 2019, 2020, 2021, 2022, 2023)

**Domains measured:**
- Work (participation, segregation, quality of work)
- Money (financial resources, economic situation)
- Knowledge (educational attainment, segregation, lifelong learning)
- Time (care activities, social activities)
- Power (political, economic, social power)
- Health (status, behavior, access)
- Plus: Violence (prevalence, disclosure, attitudes) and Intersecting inequalities

## 3) Unit of analysis
Country–year (EU27 member states with biennial index updates)

## 4) Temporal scope
**Target:** 2020–2024 (post-COVID, aligned with WBL canonical scope)
**Available:** 2020, 2021, 2022, 2023 (confirm exact years after fetch)
**Note:** Pre-2020 data excluded from canonical analysis but may be retained for reference.

## 5) Geographic scope
**Target:** EU27 countries
**Note:** EIGE index specifically designed for EU member states. Does NOT include Balkan candidate countries (unlike WBL which has broader coverage).

## 6) Expected file format
**Format:** XLSX (Excel workbook) or CSV (downloadable from EIGE Gender Statistics Database)
**Expected structure:**
- Multiple sheets (if XLSX): Overview, Domain scores, Sub-domain indicators
- Country rows × Indicator columns × Year variants
- Possible wide format (years as columns) or long format (year as row dimension)

## 7) Access method (Phase 1)
**Method:** Manual browser export from EIGE Gender Statistics Database
**Source page URL:** https://dgs-p.eige.europa.eu/data/view?code=index__index_scores
**Authentication:** None required (public data, but Cloudflare-protected)
**License:** EU Open Data License (verify terms on EIGE portal)

**Why manual mode:**
- EIGE website uses Cloudflare protection (direct HTTP download returns 403 Forbidden)
- No public API endpoint for bulk Gender Equality Index downloads
- Browser export is the only reliable method for Phase 1 raw acquisition

**Export instructions:**
1. Navigate to source page URL in browser
2. Use portal interface to export dataset (XLSX or CSV format)
3. Save to local Downloads folder
4. Use `fetch_eige.py --local-file` mode to copy into `data/raw/eige/`

**Raw storage:**
- Output folder: `data/raw/eige/`
- Raw file: `data/raw/eige/<filename>` (as downloaded, or with run_id prefix if collision)
- Provenance log: `data/raw/eige/download_log.jsonl`

**Phase 2 consideration:** Investigate EIGE API or data.europa.eu bulk downloads for reproducibility

## 8) Variables to extract (initial placeholder)
**Core indicators (to confirm after Phase 1 fetch):**
- Overall Gender Equality Index score
- Domain scores:
  - Work domain score
  - Money domain score
  - Knowledge domain score
  - Time domain score
  - Power domain score
  - Health domain score
- Sub-domain indicators (if available)
- Violence indicators (separate module)

**Canonical columns (Phase 2 target):**
- country_code (ISO 3166-1 alpha-3)
- country_name
- year
- eige_index (overall score)
- eige_work_domain
- eige_money_domain
- eige_knowledge_domain
- eige_time_domain
- eige_power_domain
- eige_health_domain

## 9) Known limitations (to verify after fetch)
- **Biennial updates only** (not annual like WBL)
- **EU27 only** (no Balkan coverage, unlike WBL)
- **Methodology revisions** (2020 update introduced methodological changes)
- **Missing data** (some sub-indicators may have gaps for specific countries/years)
- **Country naming** (check if EIGE uses ISO codes or country names)
- **Score ranges** (verify if 0-100 scale or different normalization)

## 10) Known risks
- **Availability:** Direct bulk download link may not be obvious; may require navigation through Gender Statistics Database portal
- **Format variability:** Dataset structure may differ between years (wide vs long format)
- **Metadata:** Documentation may be separate from data file
- **Revisions:** Index methodology updated in 2020; historical comparisons must account for this
- **Sub-indicator granularity:** Not all sub-domain indicators may be downloadable in bulk format

## 11) Phase 1 exit criteria checklist
- [ ] Raw EIGE file downloaded to `data/raw/eige/`
- [ ] File is non-empty and has expected extension (.xlsx or .csv)
- [ ] If XLSX: worksheet names enumerated (no cell parsing)
- [ ] download_log.jsonl created with fetch metadata (sha256, timestamp, URL, file size)
- [ ] validation_report_<run_id>.md written to data/raw/eige/
- [ ] Inspection note added to this document (section 12 below)
- [ ] NO data cleaning, transformation, or merging performed
- [ ] NO database writes or Streamlit modifications

## 12) Raw inspection note (post-fetch)

**TODO: Fill after Phase 1 fetch completes**

- File name: `<to_fill>`
- Format: `<to_fill>` (XLSX, CSV, or JSON)
- File size: `<to_fill>` MB
- Source: European Institute for Gender Equality (EIGE)
- Download method: `<to_fill>` (manual browser export)
- License: `<to_fill>` (verify from official source)

### Worksheets/Structure detected (if XLSX)
`<to_fill after fetch>`

### Initial observations
`<to_fill after fetch>`
- Number of rows/columns
- Year coverage observed
- Country coverage observed
- Domain indicators present
- Data format (wide/long)

---

**Status:** Phase 1 scaffolding complete. Awaiting fetch and validation execution.
