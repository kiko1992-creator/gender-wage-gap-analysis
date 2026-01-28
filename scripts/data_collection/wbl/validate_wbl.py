"""
Validate World Bank WBL Raw Data - PHASE 1

Phase 1: File-level and workbook structure validation only.
Does NOT parse data or enforce schemas.

Checks:
- File exists and is readable
- File extension is .xlsx
- File size > 0
- Workbook can be opened
- Worksheet inventory (names, dimensions)
- Optional: Download log reference

Usage:
    python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx
    python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx --strict

Author: PhD Research Project
Date: January 2026
"""

import argparse
import sys
import json
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(1)


class WBLValidator:
    """Validate raw WBL XLSX file (Phase 1 - file level only)."""

    def __init__(self, file_path: str, strict: bool = False):
        """
        Initialize validator.

        Args:
            file_path: Path to XLSX file
            strict: Treat warnings as errors
        """
        self.file_path = Path(file_path)
        self.strict = strict
        self.errors = []
        self.warnings = []
        self.workbook = None

    def log(self, message: str, level: str = "INFO"):
        """Print log message."""
        symbols = {"INFO": "[*]", "SUCCESS": "[+]", "ERROR": "[-]", "WARN": "[!]"}
        symbol = symbols.get(level, "[*]")
        print(f"{symbol} {message}")

    def check_file_exists(self) -> bool:
        """Check if file exists and is readable."""
        self.log("\nCheck 1: File existence...")

        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            self.log(f"  FAIL: File not found", "ERROR")
            return False

        if not self.file_path.is_file():
            self.errors.append(f"Path is not a file: {self.file_path}")
            self.log(f"  FAIL: Not a file", "ERROR")
            return False

        size_mb = self.file_path.stat().st_size / 1024 / 1024
        self.log(f"  PASS: File exists ({size_mb:.2f} MB)", "SUCCESS")
        return True

    def check_file_extension(self) -> bool:
        """Check if file has .xlsx extension."""
        self.log("\nCheck 2: File extension...")

        if self.file_path.suffix.lower() != '.xlsx':
            self.errors.append(f"Invalid extension: {self.file_path.suffix} (expected .xlsx)")
            self.log(f"  FAIL: Extension is {self.file_path.suffix}, expected .xlsx", "ERROR")
            return False

        self.log(f"  PASS: Extension is .xlsx", "SUCCESS")
        return True

    def check_file_size(self) -> bool:
        """Check if file size is greater than zero."""
        self.log("\nCheck 3: File size...")

        size_bytes = self.file_path.stat().st_size

        if size_bytes == 0:
            self.errors.append("File is empty (0 bytes)")
            self.log(f"  FAIL: File is empty", "ERROR")
            return False

        if size_bytes < 1024:
            self.warnings.append(f"File is very small ({size_bytes} bytes)")
            self.log(f"  WARN: File is only {size_bytes} bytes", "WARN")

        size_mb = size_bytes / 1024 / 1024
        self.log(f"  PASS: File size is {size_mb:.2f} MB", "SUCCESS")
        return True

    def check_workbook_opens(self) -> bool:
        """Check if workbook can be opened with openpyxl."""
        self.log("\nCheck 4: Workbook loading...")

        try:
            # Open workbook in read-only mode (Phase 1: no modifications)
            self.workbook = load_workbook(
                filename=str(self.file_path),
                read_only=True,
                data_only=True
            )
            self.log(f"  PASS: Workbook loaded successfully", "SUCCESS")
            return True
        except Exception as e:
            self.errors.append(f"Failed to load workbook: {str(e)}")
            self.log(f"  FAIL: {str(e)}", "ERROR")
            return False

    def check_worksheet_inventory(self) -> bool:
        """List all worksheets and their dimensions."""
        self.log("\nCheck 5: Worksheet inventory...")

        if self.workbook is None:
            self.errors.append("Workbook not loaded (cannot check worksheets)")
            self.log(f"  FAIL: Workbook not loaded", "ERROR")
            return False

        try:
            sheet_names = self.workbook.sheetnames
            num_sheets = len(sheet_names)

            self.log(f"  Found {num_sheets} worksheet(s):", "INFO")

            for sheet_name in sheet_names:
                sheet = self.workbook[sheet_name]

                # Get dimensions (max_row and max_column)
                max_row = sheet.max_row
                max_col = sheet.max_column

                self.log(f"    - '{sheet_name}': {max_row} rows × {max_col} columns")

                # Warn if sheet is empty
                if max_row == 0 or max_col == 0:
                    self.warnings.append(f"Sheet '{sheet_name}' appears empty")

            self.log(f"  PASS: Inventory complete", "SUCCESS")
            return True

        except Exception as e:
            self.errors.append(f"Failed to inventory worksheets: {str(e)}")
            self.log(f"  FAIL: {str(e)}", "ERROR")
            return False

    def check_download_log(self) -> bool:
        """Optional: Check if download log references this file."""
        self.log("\nCheck 6: Download log (optional)...")

        log_path = self.file_path.parent / 'download_log.jsonl'

        if not log_path.exists():
            self.log(f"  INFO: No download log found (optional check)", "INFO")
            return True

        try:
            found_reference = False
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    log_entry = json.loads(line)

                    # Check if this log entry references our file
                    if (log_entry.get('output_file') == self.file_path.name or
                        self.file_path.name in log_entry.get('output_path', '')):
                        found_reference = True
                        self.log(f"  INFO: Found log entry from {log_entry.get('timestamp')}")
                        self.log(f"    Source: {log_entry.get('source')}")
                        self.log(f"    Method: {log_entry.get('fetch_method')}")
                        self.log(f"    SHA256: {log_entry.get('sha256', 'N/A')[:16]}...")
                        break

            if found_reference:
                self.log(f"  PASS: File referenced in download log", "SUCCESS")
            else:
                self.warnings.append("File not found in download log")
                self.log(f"  WARN: File not referenced in log", "WARN")

            return True

        except Exception as e:
            self.warnings.append(f"Could not read download log: {str(e)}")
            self.log(f"  WARN: {str(e)}", "WARN")
            return True  # Non-critical check

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if self.errors:
            print(f"\n{len(self.errors)} ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n{len(self.warnings)} WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED")
        elif not self.errors:
            print(f"\n⚠️  PASSED WITH {len(self.warnings)} WARNINGS")
        else:
            print(f"\n❌ VALIDATION FAILED: {len(self.errors)} errors")

        print("\n" + "=" * 80)
        print("PHASE 1 VALIDATION")
        print("=" * 80)
        print("This is a Phase 1 (raw file) validation.")
        print("Data parsing and schema checks will happen in Phase 2.")
        print("=" * 80)

    def validate(self) -> bool:
        """Execute full validation pipeline."""
        print("=" * 80)
        print("WBL RAW DATA VALIDATION — PHASE 1")
        print("=" * 80)
        print(f"File: {self.file_path}")
        print(f"Strict mode: {'ON' if self.strict else 'OFF'}")
        print("=" * 80)
        print("\nPhase 1: File-level validation only")
        print("(No data parsing or schema enforcement)")

        # Run checks
        checks = [
            self.check_file_exists(),
            self.check_file_extension(),
            self.check_file_size(),
            self.check_workbook_opens(),
            self.check_worksheet_inventory(),
            self.check_download_log()
        ]

        # Close workbook if opened
        if self.workbook:
            try:
                self.workbook.close()
            except:
                pass

        # Summary
        self.print_summary()

        # Determine pass/fail
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if has_errors:
            return False
        elif has_warnings and self.strict:
            print("\nStrict mode: Treating warnings as errors")
            return False
        else:
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate World Bank WBL raw XLSX file (Phase 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 1 validation checks:
  - File exists and is readable
  - File extension is .xlsx
  - File size > 0
  - Workbook can be opened
  - Worksheet inventory (names and dimensions)
  - Optional: Download log reference

Phase 1 does NOT check:
  - Data schemas (country, year columns)
  - Data values or ranges
  - Duplicate keys or missingness
  (These checks happen in Phase 2)

Examples:
  python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx
  python scripts/data_collection/wbl/validate_wbl.py data/raw/wbl/wbl.xlsx --strict
        """
    )

    parser.add_argument(
        'file',
        type=str,
        help='Path to WBL XLSX file (e.g., data/raw/wbl/wbl.xlsx)'
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors (exit code 1)'
    )

    args = parser.parse_args()

    # Validate
    validator = WBLValidator(
        file_path=args.file,
        strict=args.strict
    )

    success = validator.validate()

    # Exit code
    if success:
        print("\n✅ Validation complete: PASS (Phase 1)")
        print("\nNext steps:")
        print("1. Open XLSX file to manually inspect structure")
        print("2. Document sheet names and column headers in docs/phase1_source_wbl.md")
        print("3. Phase 2: Implement schema validation and data quality checks")
        sys.exit(0)
    else:
        print("\n❌ Validation complete: FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()
