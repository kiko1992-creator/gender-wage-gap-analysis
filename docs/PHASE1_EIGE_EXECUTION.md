# EIGE Phase 1 Execution Guide (PC)

**Status:** Ready to execute on PC with internet browser
**Date created:** 2026-02-10
**Phase:** 1 of 3 (Raw acquisition only)

---

## Prerequisites

✅ Working directory: `c:\Users\User\gender-wage-gap-analysis`
✅ Python environment with dependencies installed (`pandas`, `openpyxl`)
✅ Internet browser for manual data export
✅ Scripts ready: `scripts\data_collection\eige\fetch_eige.py`, `validate_eige.py`

---

## Phase 1 Checklist (Run Later on PC)

### Step 1: Manual Browser Export

1. **Navigate to EIGE Gender Statistics Database**
   URL: `https://dgs-p.eige.europa.eu/data/view?code=index__index_scores`

2. **Export data manually**
   - Use browser interface to export Gender Equality Index scores
   - Recommended format: XLSX (preserves structure) or CSV
   - Save to Downloads folder (e.g., `C:\Users\User\Downloads\gender_equality_index.xlsx`)
   - **Note filename** for next step

3. **Verify downloaded file**
   - Check file is non-empty (> 0 bytes)
   - Note actual filename (may differ from example)

---

### Step 2: Run fetch_eige.py (Manual Mode)

**Command template:**
```cmd
py scripts\data_collection\eige\fetch_eige.py --local-file "C:\Users\User\Downloads\<ACTUAL_FILENAME>.xlsx" --source-page-url "https://dgs-p.eige.europa.eu/data/view?code=index__index_scores"
```

**Example (replace `<ACTUAL_FILENAME>` with your downloaded file):**
```cmd
py scripts\data_collection\eige\fetch_eige.py --local-file "C:\Users\User\Downloads\gender_equality_index.xlsx" --source-page-url "https://dgs-p.eige.europa.eu/data/view?code=index__index_scores"
```

**Expected output:**
```
[EIGE Fetch] Manual acquisition complete.
[EIGE Fetch] Saved: data\raw\eige\gender_equality_index.xlsx
[EIGE Fetch] SHA256: abc123def456...
[EIGE Fetch] Size: 1,234,567 bytes
[EIGE Fetch] Log: data\raw\eige\download_log.jsonl
```

**Output files:**
- `data\raw\eige\<filename>.xlsx` (raw data file)
- `data\raw\eige\download_log.jsonl` (provenance record)

---

### Step 3: Run validate_eige.py

**Command template:**
```cmd
py scripts\data_collection\eige\validate_eige.py --input-file data\raw\eige\<ACTUAL_FILENAME>.xlsx
```

**Example:**
```cmd
py scripts\data_collection\eige\validate_eige.py --input-file data\raw\eige\gender_equality_index.xlsx
```

**Expected output:**
```
[EIGE Validation] Run ID: 20260210_143025
[EIGE Validation] Validating: data\raw\eige\gender_equality_index.xlsx
[EIGE Validation] Wrote: data\raw\eige\validation_report_20260210_143025.md
[EIGE Validation] ✓ PASS - File-level validation passed
[EIGE Validation] Found 3 worksheet(s)
```

**Output files:**
- `data\raw\eige\validation_report_<run_id>.md` (validation summary)

---

### Step 4: Inspect Validation Report

**Open report:**
```cmd
notepad data\raw\eige\validation_report_<run_id>.md
```

**Check:**
- ✅ File exists and non-empty
- ✅ Valid extension (.xlsx, .csv, or .json)
- ✅ XLSX worksheets enumerated (if applicable)
- ✅ Overall status: PASS

---

### Step 5: Document Raw Inspection

**Manually inspect file:**
1. Open XLSX file in Excel or LibreOffice
2. Note worksheet names (do NOT parse cell contents)
3. Estimate row/column counts
4. Identify year coverage (from headers or visual scan)
5. Identify country coverage (from first column or visual scan)

**Update documentation:**
- Edit `docs\phase1_source_eige.md`, Section 13 "Raw inspection note"
- Fill in: filename, format, file size, worksheet names, initial observations
- Mark checklist items in Section 12 as complete

**Append to execution log:**
- Edit `docs\PHASE1_LOG.md`, add entry with date, files created, status

---

## Phase 1 Completion Criteria

Phase 1 is complete when:

- ✅ Raw EIGE file saved to `data\raw\eige\`
- ✅ File is non-empty and has expected extension
- ✅ `download_log.jsonl` created with provenance metadata
- ✅ `validation_report_<run_id>.md` written with PASS status
- ✅ Worksheet names enumerated (if XLSX)
- ✅ Inspection note added to `docs\phase1_source_eige.md` Section 13
- ✅ Execution log entry added to `docs\PHASE1_LOG.md`
- ❌ NO data cleaning, transformation, or merging performed
- ❌ NO database writes or Streamlit modifications

---

## Why URL Mode Is Not Used

**URL mode is a placeholder** because:
- EIGE website uses Cloudflare protection (403 Forbidden on direct download)
- No public API endpoint for bulk downloads
- Manual browser export is the only reliable method for Phase 1

**URL mode command (for reference only, will not download):**
```cmd
py scripts\data_collection\eige\fetch_eige.py --url "https://dgs-p.eige.europa.eu/export/gender_equality_index" --source-page-url "https://dgs-p.eige.europa.eu/data/view?code=index__index_scores"
```

This logs the URL intent but performs NO download.

---

## Phase Boundaries Reminder

**Phase 1:** Raw acquisition only (atomic write, SHA256+size, JSONL provenance)
**Phase 2:** Source-by-source normalization (EIGE → normalized CSV, no cross-source merges)
**Phase 3:** BLOCKED until WBL + EIGE are both normalized+validated (descriptive only, no causal language)

**After Phase 1 EIGE completes:**
- Proceed to Phase 2 EIGE (normalize EIGE data)
- Continue Phase 1 WBL execution (if not already complete)
- Phase 3 remains blocked until BOTH sources normalized

---

## Troubleshooting

**Issue:** `FileNotFoundError: Local file not found`
**Fix:** Check path to downloaded file, ensure no typos, use absolute path or escape spaces

**Issue:** `FileExistsError: Refusing to overwrite existing file`
**Fix:** Previous run already saved file with same name. Use `--run-id` to create unique filename:
```cmd
py scripts\data_collection\eige\fetch_eige.py --local-file "C:\Users\User\Downloads\file.xlsx" --source-page-url "https://dgs-p.eige.europa.eu/data/view?code=index__index_scores" --run-id "2026-02-10_v2"
```

**Issue:** Validation fails with "Unexpected extension"
**Fix:** Ensure downloaded file is XLSX, CSV, or JSON (not HTML or other format)

**Issue:** Validation fails with "Failed to read XLSX worksheets"
**Fix:** File may be corrupted. Re-download and try again.

---

**Next steps after Phase 1 complete:** See Phase 2 documentation (to be created)
