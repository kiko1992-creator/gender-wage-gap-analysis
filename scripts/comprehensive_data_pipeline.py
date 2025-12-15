"""
Comprehensive Gender Wage Gap Data Pipeline
============================================
Fetches, processes, and integrates data from multiple sources:
- Eurostat (EU statistics)
- World Bank (global indicators)
- ILO (labor statistics)
- Direct CSV downloads from official sources

Target Countries: North Macedonia, Serbia, Albania, Kosovo,
                  Bosnia & Herzegovina, Montenegro + EU neighbors

Author: Gender Wage Gap Analysis Project
Date: December 2025
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
import numpy as np
import requests

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class DataPipeline:
    """Comprehensive data pipeline for gender wage gap analysis."""

    def __init__(self, base_dir=None):
        """Initialize the pipeline with project directories."""
        if base_dir is None:
            # Get the project root (parent of scripts folder)
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.enriched_dir = self.data_dir / 'enriched'
        self.processed_dir = self.data_dir / 'processed'

        # Create directories if they don't exist
        for dir_path in [self.enriched_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.eurostat_base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
        self.worldbank_base = "https://api.worldbank.org/v2"
        self.ilo_base = "https://www.ilo.org/ilostat-files/WEB_bulk_download/indicator"

        # Target countries
        self.balkan_countries = {
            'MKD': 'North Macedonia',
            'SRB': 'Serbia',
            'ALB': 'Albania',
            'XKX': 'Kosovo',
            'BIH': 'Bosnia and Herzegovina',
            'MNE': 'Montenegro'
        }

        # EU neighbors for comparison
        self.eu_neighbors = {
            'BGR': 'Bulgaria',
            'HRV': 'Croatia',
            'ROU': 'Romania',
            'SVN': 'Slovenia',
            'GRC': 'Greece',
            'HUN': 'Hungary'
        }

        self.all_countries = {**self.balkan_countries, **self.eu_neighbors}

        # Data storage
        self.datasets = {}

        print("=" * 70)
        print("COMPREHENSIVE GENDER WAGE GAP DATA PIPELINE")
        print("=" * 70)
        print(f"Project directory: {self.base_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Target countries: {len(self.all_countries)}")
        print("=" * 70)

    def log(self, message, level="INFO"):
        """Print log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "[*]", "SUCCESS": "[+]", "ERROR": "[-]", "WARN": "[!]"}
        symbol = symbols.get(level, "[*]")
        print(f"{timestamp} {symbol} {message}")

    def fetch_with_retry(self, url, params=None, retries=3, timeout=30):
        """Fetch URL with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = (attempt + 1) * 5
                    self.log(f"Rate limited, waiting {wait_time}s...", "WARN")
                    time.sleep(wait_time)
                else:
                    self.log(f"HTTP {response.status_code} for {url}", "WARN")
            except requests.exceptions.Timeout:
                self.log(f"Timeout (attempt {attempt + 1}/{retries})", "WARN")
            except requests.exceptions.RequestException as e:
                self.log(f"Request error: {str(e)[:50]}", "ERROR")

            if attempt < retries - 1:
                time.sleep(2)

        return None

    # =========================================================================
    # EUROSTAT DATA FETCHING
    # =========================================================================

    def fetch_eurostat_gpg(self):
        """
        Fetch Eurostat Gender Pay Gap data.
        Dataset: earn_gr_gpgr2 (GPG by economic activity)
        """
        self.log("Fetching Eurostat Gender Pay Gap data...")

        # Direct Eurostat dataset URLs for gender pay gap
        datasets = {
            'gpg_overall': 'sdg_05_20',      # Gender pay gap (unadjusted)
            'gpg_nace': 'earn_gr_gpgr2',     # GPG by NACE Rev.2 activity
            'earnings_sex': 'earn_mw_cur',   # Monthly earnings by sex
        }

        all_data = []

        for name, code in datasets.items():
            try:
                # Try JSON-stat format first
                url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{code}"
                params = {
                    'format': 'JSON',
                    'lang': 'en'
                }

                response = self.fetch_with_retry(url, params)

                if response:
                    try:
                        data = response.json()
                        self.log(f"  Fetched {name}: {len(str(data))} bytes", "SUCCESS")
                        all_data.append({
                            'source': name,
                            'code': code,
                            'data': data
                        })
                    except json.JSONDecodeError:
                        self.log(f"  Could not parse JSON for {name}", "WARN")
                else:
                    self.log(f"  Could not fetch {name}", "WARN")

            except Exception as e:
                self.log(f"  Error fetching {name}: {str(e)[:50]}", "ERROR")

        if all_data:
            self.datasets['eurostat_gpg'] = all_data
            self.log(f"Eurostat GPG: {len(all_data)} datasets fetched", "SUCCESS")

        return all_data

    def fetch_eurostat_labor_force(self):
        """
        Fetch Eurostat labor force participation data by sex.
        """
        self.log("Fetching Eurostat labor force participation data...")

        # Labor force participation rate by sex
        url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsa_argan"
        params = {
            'format': 'JSON',
            'sex': 'F,M,T',
            'age': 'Y15-64',
            'geo': ','.join(self.all_countries.keys())
        }

        response = self.fetch_with_retry(url, params)

        if response:
            try:
                data = response.json()
                self.datasets['eurostat_lfp'] = data
                self.log("Eurostat labor force data fetched", "SUCCESS")
                return data
            except json.JSONDecodeError:
                self.log("Could not parse Eurostat LFP data", "WARN")

        return None

    # =========================================================================
    # WORLD BANK DATA FETCHING
    # =========================================================================

    def fetch_worldbank_indicators(self):
        """
        Fetch World Bank gender-related labor indicators.
        """
        self.log("Fetching World Bank indicators...")

        indicators = {
            # Labor force participation
            'SL.TLF.CACT.FE.ZS': 'Labor force participation rate, female (% of female population ages 15+)',
            'SL.TLF.CACT.MA.ZS': 'Labor force participation rate, male (% of male population ages 15+)',
            'SL.TLF.CACT.ZS': 'Labor force participation rate, total',

            # Unemployment
            'SL.UEM.TOTL.FE.ZS': 'Unemployment, female (% of female labor force)',
            'SL.UEM.TOTL.MA.ZS': 'Unemployment, male (% of male labor force)',

            # Wage and salaried workers
            'SL.EMP.WORK.FE.ZS': 'Wage and salaried workers, female (% of female employment)',
            'SL.EMP.WORK.MA.ZS': 'Wage and salaried workers, male (% of male employment)',

            # Employment in sectors
            'SL.SRV.EMPL.FE.ZS': 'Employment in services, female (% of female employment)',
            'SL.IND.EMPL.FE.ZS': 'Employment in industry, female (% of female employment)',
            'SL.AGR.EMPL.FE.ZS': 'Employment in agriculture, female (% of female employment)',

            # Education
            'SE.TER.CUAT.BA.FE.ZS': 'Educational attainment, at least Bachelor\'s, female',
            'SE.TER.CUAT.BA.MA.ZS': 'Educational attainment, at least Bachelor\'s, male',

            # Part-time employment
            'SL.TLF.PART.FE.ZS': 'Part time employment, female (% of total female employment)',
            'SL.TLF.PART.MA.ZS': 'Part time employment, male (% of total male employment)',
        }

        all_data = []
        countries_str = ';'.join(self.all_countries.keys())

        for indicator_code, description in indicators.items():
            try:
                url = f"{self.worldbank_base}/country/{countries_str}/indicator/{indicator_code}"
                params = {
                    'format': 'json',
                    'per_page': 1000,
                    'date': '2009:2024'
                }

                response = self.fetch_with_retry(url, params)

                if response:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        records = data[1]
                        self.log(f"  {indicator_code}: {len(records)} records", "SUCCESS")
                        all_data.append({
                            'indicator': indicator_code,
                            'description': description,
                            'records': records
                        })
                    else:
                        self.log(f"  {indicator_code}: No data available", "WARN")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.log(f"  Error fetching {indicator_code}: {str(e)[:40]}", "ERROR")

        if all_data:
            self.datasets['worldbank'] = all_data
            self.log(f"World Bank: {len(all_data)} indicators fetched", "SUCCESS")

        return all_data

    def convert_worldbank_to_df(self, wb_data):
        """Convert World Bank API response to pandas DataFrame."""
        rows = []

        for indicator_data in wb_data:
            for record in indicator_data.get('records', []):
                if record.get('value') is not None:
                    rows.append({
                        'country_code': record['country']['id'],
                        'country': record['country']['value'],
                        'year': int(record['date']),
                        'indicator_code': indicator_data['indicator'],
                        'indicator': indicator_data['description'],
                        'value': record['value']
                    })

        return pd.DataFrame(rows)

    # =========================================================================
    # ILO DATA FETCHING
    # =========================================================================

    def fetch_ilo_data(self):
        """
        Fetch ILO wage and employment data.
        Uses ILOSTAT bulk download files.
        """
        self.log("Fetching ILO data...")

        # ILO indicator codes for wages and gender
        ilo_indicators = {
            'EAR_4MTH_SEX_ECO_CUR_NB': 'Mean nominal monthly earnings by sex and economic activity',
            'EAR_HEES_SEX_OCU_NB': 'Mean hourly earnings by sex and occupation',
            'EMP_TEMP_SEX_AGE_NB': 'Employment by sex and age',
            'UNE_TUNE_SEX_AGE_NB': 'Unemployment by sex and age',
        }

        all_data = []

        for code, description in ilo_indicators.items():
            try:
                # ILO bulk download URL pattern
                url = f"https://www.ilo.org/ilostat-files/WEB_bulk_download/indicator/{code}.csv.gz"

                self.log(f"  Trying ILO {code}...")
                response = self.fetch_with_retry(url, timeout=60)

                if response:
                    # Decompress and read CSV
                    import gzip
                    try:
                        decompressed = gzip.decompress(response.content)
                        df = pd.read_csv(StringIO(decompressed.decode('utf-8')))

                        # Filter to our countries
                        country_codes = list(self.all_countries.keys())
                        df_filtered = df[df['ref_area'].isin(country_codes)]

                        if len(df_filtered) > 0:
                            self.log(f"  {code}: {len(df_filtered)} records", "SUCCESS")
                            all_data.append({
                                'indicator': code,
                                'description': description,
                                'data': df_filtered
                            })
                        else:
                            self.log(f"  {code}: No data for target countries", "WARN")
                    except Exception as e:
                        self.log(f"  Could not decompress {code}: {str(e)[:30]}", "WARN")
                else:
                    self.log(f"  Could not fetch {code}", "WARN")

            except Exception as e:
                self.log(f"  Error with {code}: {str(e)[:40]}", "ERROR")

        if all_data:
            self.datasets['ilo'] = all_data
            self.log(f"ILO: {len(all_data)} datasets fetched", "SUCCESS")

        return all_data

    # =========================================================================
    # DIRECT DATA DOWNLOADS
    # =========================================================================

    def fetch_direct_sources(self):
        """
        Fetch data from direct downloadable sources.
        """
        self.log("Fetching direct source data...")

        direct_sources = [
            {
                'name': 'UN Women Gender Stats',
                'url': 'https://data.unwomen.org/sites/default/files/documents/2022-10/Gender_Snapshot_Europe_Central_Asia_2022_web.pdf',
                'type': 'pdf',
                'skip': True  # PDF requires special handling
            },
            {
                'name': 'Eurostat GPG CSV',
                'url': 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/SDG_05_20/?format=CSV',
                'type': 'csv'
            }
        ]

        for source in direct_sources:
            if source.get('skip'):
                continue

            try:
                self.log(f"  Fetching {source['name']}...")
                response = self.fetch_with_retry(source['url'], timeout=60)

                if response and source['type'] == 'csv':
                    df = pd.read_csv(StringIO(response.text))
                    self.datasets[source['name']] = df
                    self.log(f"  {source['name']}: {len(df)} records", "SUCCESS")

            except Exception as e:
                self.log(f"  Error with {source['name']}: {str(e)[:40]}", "ERROR")

    # =========================================================================
    # DATA PROCESSING AND INTEGRATION
    # =========================================================================

    def load_existing_data(self):
        """Load existing local datasets."""
        self.log("Loading existing local data...")

        existing_files = [
            ('expanded_balkan_wage_data.csv', 'main_dataset'),
            ('macedonia_wage_sample.csv', 'macedonia_sample')
        ]

        for filename, key in existing_files:
            filepath = self.raw_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                self.datasets[key] = df
                self.log(f"  Loaded {filename}: {len(df)} records", "SUCCESS")
            else:
                self.log(f"  {filename} not found", "WARN")

    def process_and_save(self):
        """Process all fetched data and save to files."""
        self.log("Processing and saving data...")

        # Save World Bank data
        if 'worldbank' in self.datasets:
            wb_df = self.convert_worldbank_to_df(self.datasets['worldbank'])
            if len(wb_df) > 0:
                output_path = self.enriched_dir / 'worldbank_labor_indicators.csv'
                wb_df.to_csv(output_path, index=False)
                self.log(f"  Saved World Bank data: {len(wb_df)} records", "SUCCESS")

                # Also create a pivoted version
                pivot_df = wb_df.pivot_table(
                    index=['country', 'country_code', 'year'],
                    columns='indicator_code',
                    values='value'
                ).reset_index()
                pivot_path = self.enriched_dir / 'worldbank_indicators_wide.csv'
                pivot_df.to_csv(pivot_path, index=False)
                self.log(f"  Saved pivoted World Bank data", "SUCCESS")

        # Save ILO data
        if 'ilo' in self.datasets:
            for item in self.datasets['ilo']:
                filename = f"ilo_{item['indicator'].lower()}.csv"
                output_path = self.enriched_dir / filename
                item['data'].to_csv(output_path, index=False)
                self.log(f"  Saved ILO {item['indicator']}", "SUCCESS")

        # Save Eurostat data (if available in tabular form)
        if 'Eurostat GPG CSV' in self.datasets:
            df = self.datasets['Eurostat GPG CSV']
            output_path = self.enriched_dir / 'eurostat_gpg_raw.csv'
            df.to_csv(output_path, index=False)
            self.log(f"  Saved Eurostat GPG: {len(df)} records", "SUCCESS")

        # Create a summary file
        summary = {
            'fetch_date': datetime.now().isoformat(),
            'datasets': {},
            'total_records': 0
        }

        for key, data in self.datasets.items():
            if isinstance(data, pd.DataFrame):
                summary['datasets'][key] = len(data)
                summary['total_records'] += len(data)
            elif isinstance(data, list):
                total = sum(len(d.get('data', d.get('records', []))) for d in data if isinstance(d, dict))
                summary['datasets'][key] = total
                summary['total_records'] += total

        summary_path = self.enriched_dir / 'fetch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.log(f"Saved fetch summary to {summary_path}", "SUCCESS")

        return summary

    def create_integrated_dataset(self):
        """
        Create an integrated dataset combining all sources.
        """
        self.log("Creating integrated dataset...")

        # Start with existing main dataset
        if 'main_dataset' not in self.datasets:
            self.log("Main dataset not loaded, cannot create integrated dataset", "ERROR")
            return None

        main_df = self.datasets['main_dataset'].copy()

        # Add World Bank indicators if available
        if 'worldbank' in self.datasets:
            wb_df = self.convert_worldbank_to_df(self.datasets['worldbank'])

            # Map country names
            country_name_map = {v: v for v in self.all_countries.values()}
            country_name_map.update({
                'North Macedonia': 'North Macedonia',
                'Bosnia and Herzegovina': 'Bosnia and Herzegovina',
                'Serbia': 'Serbia'
            })

            # Pivot to wide format for merging
            wb_wide = wb_df.pivot_table(
                index=['country', 'year'],
                columns='indicator_code',
                values='value'
            ).reset_index()

            # Merge with main dataset
            main_df = main_df.merge(
                wb_wide,
                on=['country', 'year'],
                how='left'
            )

            self.log(f"  Added {len(wb_wide.columns) - 2} World Bank indicators", "SUCCESS")

        # Save integrated dataset
        output_path = self.processed_dir / 'integrated_wage_data.csv'
        main_df.to_csv(output_path, index=False)
        self.log(f"Saved integrated dataset: {len(main_df)} records, {len(main_df.columns)} columns", "SUCCESS")

        self.datasets['integrated'] = main_df

        return main_df

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run(self):
        """Run the complete data pipeline."""
        print("\n" + "=" * 70)
        print("STARTING DATA PIPELINE")
        print("=" * 70 + "\n")

        start_time = time.time()

        # Step 1: Load existing data
        self.load_existing_data()

        # Step 2: Fetch from APIs
        print("\n--- Fetching External Data ---\n")
        self.fetch_eurostat_gpg()
        self.fetch_eurostat_labor_force()
        self.fetch_worldbank_indicators()
        self.fetch_ilo_data()
        self.fetch_direct_sources()

        # Step 3: Process and save
        print("\n--- Processing Data ---\n")
        summary = self.process_and_save()

        # Step 4: Create integrated dataset
        print("\n--- Creating Integrated Dataset ---\n")
        integrated_df = self.create_integrated_dataset()

        # Summary
        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Datasets fetched: {len(self.datasets)}")
        print(f"Total records: {summary.get('total_records', 'N/A')}")
        print(f"\nOutput directories:")
        print(f"  Enriched data: {self.enriched_dir}")
        print(f"  Processed data: {self.processed_dir}")
        print("=" * 70)

        return self.datasets


def main():
    """Main entry point."""
    pipeline = DataPipeline()
    datasets = pipeline.run()

    # Print final dataset summary
    print("\n" + "=" * 70)
    print("DATASET INVENTORY")
    print("=" * 70)

    for name, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            print(f"  {name}: DataFrame with {len(data)} rows, {len(data.columns)} columns")
        elif isinstance(data, list):
            print(f"  {name}: List with {len(data)} items")
        elif isinstance(data, dict):
            print(f"  {name}: Dict with {len(data)} keys")

    print("\nNext steps:")
    print("1. Check data/enriched/ for individual source files")
    print("2. Check data/processed/integrated_wage_data.csv for merged data")
    print("3. Run the analysis notebooks with the enriched data")
    print("=" * 70)


if __name__ == '__main__':
    main()
