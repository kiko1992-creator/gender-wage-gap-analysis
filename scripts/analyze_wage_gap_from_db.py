"""
SERIOUS WORK: Analyze Gender Wage Gap from PostgreSQL Database
This script demonstrates how to automate your PhD research analysis

After you master this, you'll scale it to 27 EU countries!
"""

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("GENDER WAGE GAP ANALYSIS - PhD RESEARCH")
print("=" * 80)

# ==============================================================================
# STEP 1: Connect to Your PostgreSQL Database
# ==============================================================================
print("\n📡 Connecting to PostgreSQL database...")

try:
    # Try Unix socket (Linux/Mac)
    conn = psycopg2.connect(
        dbname="practice_db",
        user="postgres",
        host="/var/run/postgresql"
    )
    print("✅ Connected! (Unix socket)")
except:
    try:
        # Fallback to TCP/IP (Windows)
        conn = psycopg2.connect(
            dbname="practice_db",
            user="postgres",
            password="postgres",  # Change if you set a different password
            host="localhost",
            port="5432"
        )
        print("✅ Connected! (TCP/IP)")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        exit(1)

# ==============================================================================
# STEP 2: Load Data into Pandas (Like You Do for Your Streamlit App!)
# ==============================================================================
print("\n📊 Loading wage gap data from database...")

query = """
    SELECT
        w.country,
        w.year,
        w.gap_percent,
        w.unemployment,
        c.population,
        c.gdp_billions
    FROM wage_gap_practice w
    JOIN countries c ON w.country = c.name
    ORDER BY w.country, w.year DESC
"""

df = pd.read_sql_query(query, conn)
print(f"✅ Loaded {len(df)} records from database\n")
print(df.to_string(index=False))

# ==============================================================================
# STEP 3: Calculate Key Statistics (For Your PhD!)
# ==============================================================================
print("\n\n" + "=" * 80)
print("RESEARCH FINDING #1: Average Wage Gap by Country")
print("=" * 80)

avg_gap = df.groupby('country').agg({
    'gap_percent': ['mean', 'std', 'min', 'max'],
    'unemployment': 'mean',
    'population': 'first',
    'gdp_billions': 'first'
}).round(2)

print(avg_gap)

# ==============================================================================
# STEP 4: Identify Key Research Insights
# ==============================================================================
print("\n\n" + "=" * 80)
print("RESEARCH FINDING #2: Which Country Needs Most Attention?")
print("=" * 80)

worst_country = df.groupby('country')['gap_percent'].mean().idxmax()
worst_gap = df.groupby('country')['gap_percent'].mean().max()

best_country = df.groupby('country')['gap_percent'].mean().idxmin()
best_gap = df.groupby('country')['gap_percent'].mean().min()

print(f"\n🔴 WORST: {worst_country} - {worst_gap:.2f}% average wage gap")
print(f"🟢 BEST:  {best_country} - {best_gap:.2f}% average wage gap")
print(f"\n📊 Gap between best and worst: {worst_gap - best_gap:.2f} percentage points")

# ==============================================================================
# STEP 5: Trend Analysis (Are Things Getting Better or Worse?)
# ==============================================================================
print("\n\n" + "=" * 80)
print("RESEARCH FINDING #3: Wage Gap Trends (2022 → 2023)")
print("=" * 80)

for country in df['country'].unique():
    country_data = df[df['country'] == country].sort_values('year')
    if len(country_data) >= 2:
        change = country_data.iloc[-1]['gap_percent'] - country_data.iloc[0]['gap_percent']
        trend = "📈 WORSENING" if change > 0 else "📉 IMPROVING"
        print(f"\n{country}:")
        print(f"  2022: {country_data.iloc[0]['gap_percent']}%")
        print(f"  2023: {country_data.iloc[-1]['gap_percent']}%")
        print(f"  Change: {change:+.2f}% {trend}")

# ==============================================================================
# STEP 6: Correlation Analysis (PhD Level!)
# ==============================================================================
print("\n\n" + "=" * 80)
print("RESEARCH FINDING #4: Does Unemployment Affect Wage Gap?")
print("=" * 80)

correlation = df['gap_percent'].corr(df['unemployment'])
print(f"\nCorrelation between wage gap and unemployment: {correlation:.3f}")

if abs(correlation) > 0.7:
    relationship = "STRONG"
elif abs(correlation) > 0.4:
    relationship = "MODERATE"
else:
    relationship = "WEAK"

direction = "positive" if correlation > 0 else "negative"
print(f"Interpretation: {relationship} {direction} relationship")
print(f"\nThis means: When unemployment {'increases' if correlation > 0 else 'decreases'},")
print(f"            the wage gap tends to {'increase' if correlation > 0 else 'decrease'} as well.")

# ==============================================================================
# STEP 7: Economic Context (GDP vs Wage Gap)
# ==============================================================================
print("\n\n" + "=" * 80)
print("RESEARCH FINDING #5: Economic Development vs Gender Equality")
print("=" * 80)

country_summary = df.groupby('country').agg({
    'gap_percent': 'mean',
    'gdp_billions': 'first',
    'population': 'first'
}).round(2)

country_summary['gdp_per_capita'] = (
    country_summary['gdp_billions'] * 1000000000 / country_summary['population']
).round(0)

print("\nGDP per Capita vs Wage Gap:")
print(country_summary[['gap_percent', 'gdp_per_capita']].sort_values('gdp_per_capita'))

gdp_correlation = country_summary['gap_percent'].corr(country_summary['gdp_per_capita'])
print(f"\nCorrelation: {gdp_correlation:.3f}")
print("Interesting finding for your PhD! 💡")

# ==============================================================================
# STEP 8: Generate Summary for PhD Thesis
# ==============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY FOR YOUR PhD THESIS")
print("=" * 80)

print(f"""
📚 KEY FINDINGS FROM BALKAN REGION ANALYSIS:

1. Sample Size: {len(df)} observations across {df['country'].nunique()} countries

2. Average Wage Gap: {df['gap_percent'].mean():.2f}%

3. Range: {df['gap_percent'].min():.2f}% to {df['gap_percent'].max():.2f}%

4. Worst Performer: {worst_country} ({worst_gap:.2f}%)

5. Best Performer: {best_country} ({best_gap:.2f}%)

6. Economic Factor: {relationship} correlation between unemployment and wage gap

7. Temporal Trend: Data covers {df['year'].min()} to {df['year'].max()}

📊 NEXT STEPS FOR RESEARCH:
   → Expand to all 27 EU countries
   → Add more years of data (2015-2024)
   → Include additional variables (education, sector, age)
   → Apply Oaxaca-Blinder decomposition
   → Run panel regression with fixed effects
   → Compare EU vs non-EU Balkan countries
""")

# ==============================================================================
# STEP 9: Save Results to CSV (For Your Records)
# ==============================================================================
print("\n💾 Saving results to CSV...")
output_file = "output/balkan_wage_gap_analysis.csv"
df.to_csv(output_file, index=False)
print(f"✅ Saved to: {output_file}")

# Close database connection
conn.close()
print("\n🔒 Database connection closed")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! 🎓")
print("=" * 80)
print("\n📝 You just automated PhD-level research analysis!")
print("   This same script can analyze 27 countries × 10 years = 270 data points!")
print("\n🚀 Next: Fetch REAL data from Eurostat API and scale this up!")
print("=" * 80)
