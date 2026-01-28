# WBL Validator Fixed - Phase 1 Compliant

**Date:** 2026-01-28
**Status:** ✅ Validator rewritten, ready to test

---

## What Was Fixed

### Problem

The original validator tried to:
- Parse XLSX into pandas DataFrame
- Check for specific columns (country, year, etc.)
- Validate data values and schemas
- Crashed with `AttributeError: 'WBLValidator' object has no attribute 'year_col'`

**This was incorrect for Phase 1** because Phase 1 is raw ingestion only - files are saved as-is, not parsed.

### Solution

Completely rewrote validator to be **Phase 1 compliant**:
- Uses `openpyxl` instead of `pandas` (workbook-level inspection only)
- Checks file and workbook structure only
- Does NOT parse data or enforce schemas
- Never crashes on missing columns

---

## Phase 1 Validation Checks

The validator now checks:

1. ✅ **File exists** and is readable
2. ✅ **File extension** is `.xlsx`
3. ✅ **File size** > 0 bytes
4. ✅ **Workbook opens** with `openpyxl`
5. ✅ **Worksheet inventory** (lists all sheet names and dimensions)
6. ✅ **Download log** (optional - checks if file is referenced in `download_log.jsonl`)

### What It Does NOT Check

❌ Data schemas (country, year columns)
❌ Data values or ranges
❌ Duplicate keys
❌ Missingness
❌ Year scope or country coverage

**Rationale:** These are Phase 2 checks (after data is parsed and transformed).

---

## Example Output

### Successful Validation

```
================================================================================
WBL RAW DATA VALIDATION — PHASE 1
================================================================================
File: data\raw\wbl\wbl.xlsx
Strict mode: OFF
================================================================================

Phase 1: File-level validation only
(No data parsing or schema enforcement)

Check 1: File existence...
  [+] PASS: File exists (12.45 MB)

Check 2: File extension...
  [+] PASS: Extension is .xlsx

Check 3: File size...
  [+] PASS: File size is 12.45 MB

Check 4: Workbook loading...
  [+] PASS: Workbook loaded successfully

Check 5: Worksheet inventory...
  [*] Found 11 worksheet(s):
    - 'WBL2024': 190 rows × 15 columns
    - 'Historical-Panel': 9500 rows × 50 columns
    - 'Metadata': 45 rows × 4 columns
    - 'Indicators': 35 rows × 3 columns
    - 'Mobility': 190 rows × 8 columns
    - 'Workplace': 190 rows × 8 columns
    - 'Pay': 190 rows × 8 columns
    - 'Marriage': 190 rows × 8 columns
    - 'Parenthood': 190 rows × 8 columns
    - 'Entrepreneurship': 190 rows × 8 columns
    - 'Assets': 190 rows × 8 columns
  [+] PASS: Inventory complete

Check 6: Download log (optional)...
  [*] INFO: Found log entry from 2026-01-28T15:30:00Z
    Source: WBL
    Method: bulk_xlsx_download
    SHA256: a1b2c3d4e5f6g7h8...
  [+] PASS: File referenced in download log

================================================================================
VALIDATION SUMMARY
================================================================================

✅ ALL CHECKS PASSED

================================================================================
PHASE 1 VALIDATION
================================================================================
This is a Phase 1 (raw file) validation.
Data parsing and schema checks will happen in Phase 2.
================================================================================

✅ Validation complete: PASS (Phase 1)

Next steps:
1. Open XLSX file to manually inspect structure
2. Document sheet names and column headers in docs/phase1_source_wbl.md
3. Phase 2: Implement schema validation and data quality checks
```

### Failed Validation (File Missing)

```
================================================================================
WBL RAW DATA VALIDATION — PHASE 1
================================================================================
File: data\raw\wbl\missing.xlsx
Strict mode: OFF
================================================================================

Phase 1: File-level validation only
(No data parsing or schema enforcement)

Check 1: File existence...
  [-] FAIL: File not found

================================================================================
VALIDATION SUMMARY
================================================================================

1 ERRORS:
  1. File not found: data\raw\wbl\missing.xlsx

❌ VALIDATION FAILED: 1 errors

================================================================================
PHASE 1 VALIDATION
================================================================================
This is a Phase 1 (raw file) validation.
Data parsing and schema checks will happen in Phase 2.
================================================================================

❌ Validation complete: FAIL
```

---

## Usage

### Basic Validation

```bash
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx
```

### Strict Mode (Warnings = Errors)

```bash
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx --strict
```

### Exit Codes

- **0:** Validation passed (all checks successful)
- **1:** Validation failed (errors found)

---

## Key Changes Made

### Removed

❌ `pandas` dependency for validation
❌ `pd.read_csv()` / `pd.read_excel()` calls
❌ DataFrame column checks (country, year, etc.)
❌ Data value validation (year range, wage gap range)
❌ Duplicate key detection
❌ Missingness checks
❌ Country coverage analysis
❌ `year_col`, `country_col` attributes (caused AttributeError)

### Added

✅ `openpyxl` for workbook-level inspection
✅ `load_workbook()` with `read_only=True` (no modifications)
✅ Worksheet inventory (sheet names and dimensions)
✅ Download log reference check (optional, non-critical)
✅ Clear Phase 1 boundary documentation
✅ Proper error handling (never crashes)
✅ Workbook cleanup (`workbook.close()`)

---

## Integration with Workflow

### Phase 1 Workflow

```bash
# 1. Fetch WBL data
python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024

# 2. Validate (Phase 1 - file level)
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx

# 3. Manually inspect XLSX
# Open data/raw/wbl/wbl.xlsx in Excel/LibreOffice

# 4. Document findings
# Edit docs/phase1_source_wbl.md section 9 with:
#   - Sheet names found
#   - Columns in main data sheet
#   - Year range observed
#   - Country identifiers used
```

### Phase 2 (Future)

Phase 2 validation will:
- Parse XLSX into DataFrames (one per relevant sheet)
- Check schemas (country, year, wbl_index columns)
- Validate data values (year range 2020-2024)
- Check for duplicates and missingness
- Verify target country coverage (EU27 + Balkans)

---

## Dependencies

**Required:**
```bash
pip install openpyxl
```

**Not required for Phase 1:**
- `pandas` (not used in validation)
- `requests` (not used in validation)

---

## Testing Checklist

Before committing:

- [ ] Run validator on actual downloaded file
- [ ] Verify it passes with exit code 0
- [ ] Verify worksheet inventory is printed
- [ ] Test with missing file (should fail gracefully)
- [ ] Test with non-XLSX file (should fail with clear error)
- [ ] Test `--strict` mode
- [ ] Verify no AttributeError crashes
- [ ] Verify no pandas imports in validation code

---

## Summary

**Status:** ✅ **VALIDATOR FIXED AND READY**

**Changed file:** `scripts/data_collection/wbl/validate_wbl.py`

**Lines changed:** Entire file rewritten (~340 lines)

**Key improvement:**
- Phase 1 compliant (file-level checks only)
- Never crashes on missing columns
- Uses openpyxl (not pandas)
- Clear Phase 1 vs Phase 2 boundary

**Ready to test:**
```bash
python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx
```

**No commits made.** Changes ready for user review and testing.
