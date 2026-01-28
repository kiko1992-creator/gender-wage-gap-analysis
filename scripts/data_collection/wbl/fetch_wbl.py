"""
Fetch World Bank Women, Business & Law (WBL) Data

Phase 1: Raw data ingestion only (no cleaning, no merging)

Usage:
    python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --out data/raw/wbl/wbl.xlsx
    python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --refresh

Output:
    - Raw XLSX file with WBL indicators
    - download_log.jsonl with fetch metadata

Author: PhD Research Project
Date: January 2026
"""

import argparse
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pandas as pd
    import requests
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install pandas requests openpyxl")
    sys.exit(1)

from scripts.data_collection._shared.io import (
    write_jsonl,
    sha256_file,
    ensure_dir,
    timestamp_utc
)


# ============================================================================
# CONFIGURATION: World Bank WBL API
# ============================================================================
# TODO: PASTE CORRECT WBL ENDPOINT DETAILS HERE FROM CLAUDE APP RESEARCH
#
# The World Bank has multiple data access methods:
# 1. World Development Indicators API (api.worldbank.org/v2/)
# 2. Gender Data Portal API (genderdata.worldbank.org)
# 3. WBL Historical Dataset (bulk download)
#
# PLACEHOLDER CONFIGURATION:
# Once you determine the correct endpoint, replace this section with:
# - Base URL
# - Indicator codes (e.g., for each WBL pillar)
# - Required parameters (country codes, date ranges, format)
# - Any API keys or authentication needed
# ============================================================================

WBL_CONFIG = {
    'source_name': 'World Bank Women Business and Law',
    'source_abbr': 'WBL',
    'base_url': 'PLACEHOLDER - NEEDS RESEARCH',
    'api_version': 'PLACEHOLDER',
    'indicator_codes': {
        # Example structure (replace with actual WBL indicator codes):
        # 'pillar_mobility': 'SG.LAW.XXXX',
        # 'pillar_workplace': 'SG.LAW.YYYY',
        # 'pillar_pay': 'SG.LAW.ZZZZ',
        # ... (8 pillars total in WBL)
    },
    'bulk_download_url': 'https://wbl.worldbank.org/content/dam/sites/wbl/documents/2024/WBL2024-1-0-Historical-Panel-Data.xlsx',
    'documentation_url': 'https://wbl.worldbank.org/',
    'license': 'CC BY 4.0',
    'notes': 'WBL measures legal gender equality across 8 areas: Mobility, Workplace, Pay, Marriage, Parenthood, Entrepreneurship, Assets, Pension'
}

# Target countries (EU27 + Balkans)
TARGET_COUNTRIES = [
    # EU27
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST',
    'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA',
    'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK',
    'SVN', 'ESP', 'SWE',
    # Balkans
    'MKD',  # North Macedonia
    'SRB',  # Serbia
    'MNE',  # Montenegro
    'ALB',  # Albania
    'BIH',  # Bosnia-Herzegovina
    'XKX',  # Kosovo
]


class WBLFetcher:
    """Fetch WBL data from World Bank sources."""

    def __init__(self, years: str, output_path: str, refresh: bool = False):
        """
        Initialize WBL fetcher.

        Args:
            years: Year range (e.g., '2020-2024')
            output_path: Path to save XLSX file
            refresh: Force re-download even if file exists
        """
        self.years = self._parse_years(years)
        self.output_path = Path(output_path)
        self.refresh = refresh
        self.log_path = self.output_path.parent / 'download_log.jsonl'

        ensure_dir(self.output_path.parent)

    def _parse_years(self, years_str: str) -> list:
        """Parse year range string to list of years."""
        if '-' in years_str:
            start, end = map(int, years_str.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(years_str)]

    def _check_existing(self) -> bool:
        """Check if output file already exists."""
        if self.output_path.exists() and not self.refresh:
            print(f"⚠️  Output file exists: {self.output_path}")
            print("   Use --refresh to force re-download")
            return True
        return False

    def fetch_wbl_data(self) -> pd.DataFrame:
        """
        Fetch WBL data from World Bank (bulk XLSX download).

        Phase 1: Downloads raw XLSX file as-is, does NOT parse or transform.

        Returns:
            DataFrame with download metadata (1 row)
        """
        print("\n" + "=" * 80)
        print("WBL DATA FETCH - BULK XLSX DOWNLOAD")
        print("=" * 80)

        download_url = WBL_CONFIG['bulk_download_url']
        print(f"\n🌐 Source: {download_url}")
        print(f"📁 Target: {self.output_path}")

        # Download with streaming for memory efficiency
        print("\n⬇️  Downloading...")
        try:
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            raise

        print(f"   ✅ HTTP {response.status_code}")

        # Get content length for progress
        total_bytes = int(response.headers.get('content-length', 0))
        print(f"   📦 Size: {total_bytes / 1024 / 1024:.2f} MB")

        # Atomic download: write to temp file, then os.replace()
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.xlsx',
            dir=self.output_path.parent,
            prefix='.wbl_download_'
        )

        try:
            # Write to temp file
            bytes_written = 0
            chunk_size = 8192

            with os.fdopen(temp_fd, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            print(f"   ✅ Downloaded {bytes_written / 1024 / 1024:.2f} MB")

            # Atomic move to final location
            os.replace(temp_path, self.output_path)
            print(f"   ✅ Saved to: {self.output_path}")

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            print(f"❌ Save failed: {e}")
            raise

        # Compute file hash for integrity verification
        print("\n🔐 Computing SHA256...")
        file_hash = sha256_file(self.output_path)
        print(f"   ✅ {file_hash[:16]}...")

        # Log download metadata immediately
        self._log_download_metadata(
            source_url=download_url,
            http_status=response.status_code,
            bytes_downloaded=bytes_written,
            sha256_hash=file_hash
        )

        # Return metadata DataFrame (1 row)
        # This keeps the script structure intact without parsing the XLSX
        metadata_df = pd.DataFrame([{
            'source': WBL_CONFIG['source_abbr'],
            'source_url': download_url,
            'output_path': str(self.output_path),
            'sha256': file_hash,
            'bytes': bytes_written,
            'http_status': response.status_code,
            'years_requested': f"{min(self.years)}-{max(self.years)}",
            'timestamp': timestamp_utc(),
            'phase': 'Phase 1: Raw download only (not parsed)'
        }])

        print("\n✅ Bulk download complete")
        print("   ⚠️  XLSX file saved as-is (not opened or parsed)")
        print("   ⚠️  Use validate_wbl.py to inspect contents")

        return metadata_df

    def _log_download_metadata(self, source_url: str, http_status: int,
                                bytes_downloaded: int, sha256_hash: str) -> None:
        """
        Log download metadata to JSONL immediately after download.

        Args:
            source_url: Download URL
            http_status: HTTP response code
            bytes_downloaded: File size in bytes
            sha256_hash: SHA256 hash of downloaded file
        """
        print(f"\n📋 Logging to: {self.log_path}")

        log_entry = {
            'timestamp': timestamp_utc(),
            'source': WBL_CONFIG['source_abbr'],
            'source_full': WBL_CONFIG['source_name'],
            'fetch_method': 'bulk_xlsx_download',
            'source_url': source_url,
            'output_file': str(self.output_path.name),
            'output_path': str(self.output_path.absolute()),
            'sha256': sha256_hash,
            'bytes': bytes_downloaded,
            'http_status': http_status,
            'years_requested': f"{min(self.years)}-{max(self.years)}",
            'years_list': self.years,
            'countries_targeted': len(TARGET_COUNTRIES),
            'target_countries': TARGET_COUNTRIES,
            'license': WBL_CONFIG['license'],
            'notes': 'Phase 1: Raw XLSX download only, no parsing or transformation'
        }

        write_jsonl(self.log_path, log_entry)
        print(f"   ✅ Download logged")

    def save_xlsx(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to XLSX.

        Note: For bulk download method, this is skipped (file already saved).

        Args:
            df: DataFrame to save
        """
        # Check if this is metadata DataFrame (bulk download already saved file)
        if 'phase' in df.columns and 'Phase 1' in df['phase'].iloc[0]:
            print(f"\n📝 File already saved during bulk download (skipping)")
            return

        print(f"\n📝 Saving to: {self.output_path}")

        # Save to XLSX
        df.to_excel(self.output_path, index=False, engine='openpyxl')

        print(f"   ✅ Saved {len(df)} rows, {len(df.columns)} columns")

    def log_download(self, df: pd.DataFrame, fetch_method: str) -> None:
        """
        Log download metadata to JSONL.

        Note: For bulk download, logging is done immediately in fetch_wbl_data().
        This method is kept for compatibility but skips if already logged.

        Args:
            df: Downloaded DataFrame (or metadata DataFrame)
            fetch_method: Description of fetch method used
        """
        # Check if this is metadata DataFrame (bulk download already logged)
        if 'phase' in df.columns and 'Phase 1' in df['phase'].iloc[0]:
            print(f"\n📋 Download already logged during fetch (skipping)")
            return

        print(f"\n📋 Logging to: {self.log_path}")

        log_entry = {
            'timestamp': timestamp_utc(),
            'source': WBL_CONFIG['source_abbr'],
            'source_full': WBL_CONFIG['source_name'],
            'fetch_method': fetch_method,
            'url': WBL_CONFIG['base_url'],
            'output_file': str(self.output_path.name),
            'output_path': str(self.output_path),
            'sha256': sha256_file(self.output_path),
            'years_requested': f"{min(self.years)}-{max(self.years)}",
            'years_list': self.years,
            'countries_targeted': len(TARGET_COUNTRIES),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'license': WBL_CONFIG['license'],
            'notes': 'Phase 1: Raw fetch only, no cleaning'
        }

        write_jsonl(self.log_path, log_entry)
        print(f"   ✅ Download logged")

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary of fetched data."""
        print("\n" + "=" * 80)
        print("FETCH SUMMARY")
        print("=" * 80)

        # Check if this is metadata DataFrame (bulk download)
        if 'phase' in df.columns and 'Phase 1' in df['phase'].iloc[0]:
            # Metadata summary
            metadata = df.iloc[0]
            print(f"\n🌐 Source: {metadata['source']}")
            print(f"📦 Method: Bulk XLSX download")
            print(f"💾 Size: {metadata['bytes'] / 1024 / 1024:.2f} MB")
            print(f"🔐 SHA256: {metadata['sha256'][:16]}...")
            print(f"📅 Years requested: {metadata['years_requested']}")
            print(f"✅ HTTP Status: {metadata['http_status']}")
        else:
            # Regular DataFrame summary
            print(f"\n📊 Data shape: {len(df)} rows × {len(df.columns)} columns")

            if 'year' in df.columns:
                years_found = sorted(df['year'].unique())
                print(f"📅 Years: {years_found}")

            if 'country_code' in df.columns:
                countries_found = sorted(df['country_code'].unique())
                print(f"🌍 Countries: {len(countries_found)}")
                print(f"   {countries_found[:10]}{'...' if len(countries_found) > 10 else ''}")

        print(f"\n📁 Output: {self.output_path}")
        print(f"📝 Log: {self.log_path}")

        print("\n✅ Phase 1 fetch complete!")
        print("\n⚠️  Raw XLSX saved as-is (not parsed or transformed)")
        print("\nNext steps:")
        print(f"1. Validate: python scripts/data_collection/wbl/validate_wbl.py {self.output_path}")
        print("2. Inspect: Open XLSX file to review raw data")
        print("3. Document: Fill inspection notes in docs/phase1_source_wbl.md")

        print("=" * 80)

    def run(self) -> None:
        """Execute full fetch pipeline."""
        print("=" * 80)
        print("WORLD BANK WBL FETCH - PHASE 1")
        print("=" * 80)
        print(f"Years: {min(self.years)}-{max(self.years)}")
        print(f"Output: {self.output_path}")
        print(f"Refresh: {self.refresh}")
        print("=" * 80)

        # Check if already downloaded
        if self._check_existing():
            print("\n✅ Using existing file (use --refresh to re-download)")
            return

        # Fetch data
        print("\n🌐 Fetching WBL data...")
        try:
            df = self.fetch_wbl_data()
        except NotImplementedError as e:
            print(f"\n❌ {e}")
            sys.exit(1)

        # Save to XLSX (skipped for bulk download - already saved)
        self.save_xlsx(df)

        # Log download (skipped for bulk download - already logged)
        self.log_download(df, fetch_method='bulk_xlsx_download')

        # Print summary
        self.print_summary(df)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch World Bank Women, Business & Law (WBL) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch canonical scope (2020-2024)
  python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024

  # Fetch with custom output path
  python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --out data/raw/wbl/wbl.xlsx

  # Force re-download
  python scripts/data_collection/wbl/fetch_wbl.py --years 2020-2024 --refresh

  # Fetch extended range
  python scripts/data_collection/wbl/fetch_wbl.py --years 2015-2024
        """
    )

    parser.add_argument(
        '--years',
        type=str,
        required=True,
        help='Year range (e.g., 2020-2024 or single year 2023)',
        metavar='YYYY-YYYY'
    )

    parser.add_argument(
        '--out',
        type=str,
        default='data/raw/wbl/wbl.xlsx',
        help='Output XLSX path (default: data/raw/wbl/wbl.xlsx)',
        metavar='PATH'
    )

    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force re-download even if file exists'
    )

    args = parser.parse_args()

    # Run fetcher
    fetcher = WBLFetcher(
        years=args.years,
        output_path=args.out,
        refresh=args.refresh
    )

    fetcher.run()


if __name__ == '__main__':
    main()
