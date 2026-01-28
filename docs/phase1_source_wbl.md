# Phase 1 Source Contract — World Bank Women, Business & Law (WBL)

## 0) Hard Stop (Phase 1 boundary)
Phase 1 ends after:
- raw WBL data is fetched and saved in `data/raw/wbl/`
- an inspection note is written (rows/cols/years/country coverage)
No cleaning, no merging, no database writes, no Streamlit changes.

## 1) Purpose in this project
Legal equality framework to contextualize wage-gap outcomes (explanatory layer).

## 2) Unit of analysis
Country–year (or country with periodic updates; confirm after fetch).

## 3) Temporal scope
Canonical scope: 2020–2024 (post-COVID). Pre-2020 excluded from canonical analysis.

## 4) Geographic scope
Target: EU27 + selected Balkan countries (confirm coverage after fetch).

## 5) Access method (Phase 1)
Manual download/export from official WBL source (Phase 1). Phase 2 may convert to API.

## 6) Raw storage contract
Folder: `data/raw/wbl/`
Naming: `wbl_raw_<YYYY-MM-DD>.<ext>`
Never overwrite previous raw files.

## 7) Variables to extract (initial placeholder)
(To fill after we see the raw columns.)

## 8) Known limitations (fill after fetch)
Missingness, revisions, country naming differences, year availability, license notes.

## 9) Raw inspection note (fill after fetch)
- File name:
- Format (csv/xlsx/json):
- Rows / columns:
- Year coverage found:
- Country identifiers found (iso2/iso3/names):
- Candidate variable columns:
