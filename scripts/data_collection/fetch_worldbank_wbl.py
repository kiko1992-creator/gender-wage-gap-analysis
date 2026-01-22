"""
Fetch World Bank Women, Business and the Law (WBL) Data
========================================================

This script fetches gender equality legal framework data from the World Bank.

WBL Index measures legal equality across 8 indicators:
1. Mobility - Can women travel/work like men?
2. Workplace - Are there gender-based restrictions?
3. Pay - Equal remuneration laws?
4. Marriage - Equal rights in marriage?
5. Parenthood - Equal parental leave/childcare?
6. Entrepreneurship - Equal business rights?
7. Assets - Equal property rights?
8. Pension - Equal pension rights?

Score: 0-100 (100 = full legal equality)
"""

import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime

# Configuration
BASE_URL = "https://api.worldbank.org/v2"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "world_bank"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EU27 + Balkans country codes (ISO 3166-1 alpha-2)
COUNTRIES = {
    # EU27
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia',
    'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia',
    'FI': 'Finland', 'FR': 'France', 'DE': 'Germany', 'GR': 'Greece',
    'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia',
    'SI': 'Slovenia', 'ES': 'Spain', 'SE': 'Sweden',
    # Balkans (non-EU)
    'AL': 'Albania', 'BA': 'Bosnia and Herzegovina', 'XK': 'Kosovo',
    'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia'
}

# World Bank 3-letter codes (they use different format)
WB_COUNTRY_CODES = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'HR': 'HRV', 'CY': 'CYP',
    'CZ': 'CZE', 'DK': 'DNK', 'EE': 'EST', 'FI': 'FIN', 'FR': 'FRA',
    'DE': 'DEU', 'GR': 'GRC', 'HU': 'HUN', 'IE': 'IRL', 'IT': 'ITA',
    'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX', 'MT': 'MLT', 'NL': 'NLD',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK', 'SI': 'SVN',
    'ES': 'ESP', 'SE': 'SWE', 'AL': 'ALB', 'BA': 'BIH', 'XK': 'XKX',
    'ME': 'MNE', 'MK': 'MKD', 'RS': 'SRB'
}

# WBL Indicator codes
WBL_INDICATORS = {
    'SG.LAW.INDX': 'WBL_Index_Overall',
    'SG.LAW.NODC.MO': 'WBL_Mobility',
    'SG.LAW.NODC.WK': 'WBL_Workplace',
    'SG.LAW.NODC.PA': 'WBL_Pay',
    'SG.LAW.NODC.MA': 'WBL_Marriage',
    'SG.LAW.NODC.PR': 'WBL_Parenthood',
    'SG.LAW.NODC.EN': 'WBL_Entrepreneurship',
    'SG.LAW.NODC.AS': 'WBL_Assets',
    'SG.LAW.NODC.PE': 'WBL_Pension'
}


def fetch_wbl_data(start_year=2020, end_year=2024):
    """
    Fetch WBL data from World Bank API

    Args:
        start_year: First year to fetch
        end_year: Last year to fetch

    Returns:
        pandas DataFrame with WBL indicators
    """
    print(f"📊 Fetching World Bank Women, Business and the Law data ({start_year}-{end_year})")
    print(f"Countries: {len(COUNTRIES)}")
    print(f"Indicators: {len(WBL_INDICATORS)}")

    all_data = []

    for iso2, country_name in COUNTRIES.items():
        wb_code = WB_COUNTRY_CODES.get(iso2)
        if not wb_code:
            print(f"⚠️  Skipping {country_name} - no World Bank code mapping")
            continue

        print(f"\n🔄 Fetching data for {country_name} ({iso2}/{wb_code})...")

        for indicator_code, indicator_name in WBL_INDICATORS.items():
            try:
                # World Bank API format
                url = f"{BASE_URL}/country/{wb_code}/indicator/{indicator_code}"
                params = {
                    'format': 'json',
                    'date': f'{start_year}:{end_year}',
                    'per_page': 500
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                # World Bank returns [metadata, data]
                if len(data) > 1 and data[1]:
                    for entry in data[1]:
                        if entry['value'] is not None:
                            all_data.append({
                                'country_code': iso2,
                                'country_name': country_name,
                                'wb_country_code': wb_code,
                                'year': int(entry['date']),
                                'indicator_code': indicator_code,
                                'indicator_name': indicator_name,
                                'value': float(entry['value'])
                            })
                    print(f"  ✅ {indicator_name}: {len([e for e in data[1] if e['value'] is not None])} values")
                else:
                    print(f"  ⚠️  {indicator_name}: No data")

            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error fetching {indicator_name}: {e}")
            except Exception as e:
                print(f"  ❌ Unexpected error for {indicator_name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("\n❌ No data fetched!")
        return df

    print(f"\n✅ Successfully fetched {len(df)} data points")
    print(f"   Countries: {df['country_code'].nunique()}")
    print(f"   Years: {sorted(df['year'].unique())}")
    print(f"   Indicators: {df['indicator_name'].nunique()}")

    return df


def pivot_to_wide_format(df):
    """
    Pivot data from long to wide format (one row per country-year)

    Args:
        df: Long format DataFrame

    Returns:
        Wide format DataFrame
    """
    if df.empty:
        return df

    df_wide = df.pivot_table(
        index=['country_code', 'country_name', 'year'],
        columns='indicator_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    return df_wide


def save_data(df, df_wide):
    """Save data in both long and wide formats"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Long format
    long_path = OUTPUT_DIR / f"wbl_long_{timestamp}.csv"
    df.to_csv(long_path, index=False)
    print(f"\n💾 Saved long format: {long_path}")

    # Wide format
    wide_path = OUTPUT_DIR / f"wbl_wide_{timestamp}.csv"
    df_wide.to_csv(wide_path, index=False)
    print(f"💾 Saved wide format: {wide_path}")

    # Also save latest version (for easy access)
    latest_long = OUTPUT_DIR / "wbl_long_latest.csv"
    latest_wide = OUTPUT_DIR / "wbl_wide_latest.csv"

    df.to_csv(latest_long, index=False)
    df_wide.to_csv(latest_wide, index=False)
    print(f"💾 Saved latest versions (for easy reference)")

    # Save metadata
    metadata = {
        'fetch_date': datetime.now().isoformat(),
        'source': 'World Bank Women, Business and the Law API',
        'source_url': 'https://wbl.worldbank.org/',
        'api_endpoint': BASE_URL,
        'years': f"{df['year'].min()}-{df['year'].max()}" if not df.empty else 'N/A',
        'countries': len(df['country_code'].unique()) if not df.empty else 0,
        'indicators': list(WBL_INDICATORS.values()),
        'total_datapoints': len(df)
    }

    metadata_path = OUTPUT_DIR / f"wbl_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")

    return long_path, wide_path


def main():
    """Main execution"""
    print("="*70)
    print("WORLD BANK WOMEN, BUSINESS AND THE LAW DATA FETCHER")
    print("="*70)

    # Fetch data
    df_long = fetch_wbl_data(start_year=2020, end_year=2024)

    if df_long.empty:
        print("\n❌ Failed to fetch data. Please check your internet connection and try again.")
        return

    # Pivot to wide format
    print("\n🔄 Converting to wide format...")
    df_wide = pivot_to_wide_format(df_long)

    # Save
    save_data(df_long, df_wide)

    # Display sample
    print("\n📋 Sample data (wide format):")
    print(df_wide.head())

    print("\n📊 Summary statistics:")
    print(df_wide.describe())

    print("\n" + "="*70)
    print("✅ DATA FETCH COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the data in data/raw/world_bank/")
    print("2. Check for missing values")
    print("3. Run data cleaning script (to be created)")
    print("4. Load into PostgreSQL database")


if __name__ == "__main__":
    main()
