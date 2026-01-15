"""
INTERACTIVE PostgreSQL Practice
MODIFY THE QUERIES BELOW AND RUN THIS SCRIPT TO SEE RESULTS!

Run: sudo python scripts/practice_queries_interactive.py
"""

import psycopg2
import pandas as pd

# Connect to database
conn = psycopg2.connect(
    dbname="practice_db",
    user="postgres",
    host="/var/run/postgresql"
)
cur = conn.cursor()

print("=" * 70)
print("INTERACTIVE POSTGRESQL PRACTICE")
print("=" * 70)

# ============================================================
# EXERCISE 1: Filter countries by population
# ============================================================
print("\n📊 EXERCISE 1: Countries with population > 3 million")
print("-" * 70)

# 👉 TRY CHANGING THIS NUMBER:
min_population = 3000000  # Change this to 2000000 or 7000000 and see what happens!

cur.execute(f"SELECT name, population FROM countries WHERE population > {min_population}")
results = cur.fetchall()

for row in results:
    print(f"  {row[0]:20} - Population: {row[1]:,}")

print(f"\n💡 Found {len(results)} countries with population > {min_population:,}")


# ============================================================
# EXERCISE 2: Get wage gap data for specific country
# ============================================================
print("\n\n💰 EXERCISE 2: Wage gap data for specific country")
print("-" * 70)

# 👉 TRY CHANGING THIS COUNTRY:
country_name = 'Serbia'  # Try 'Bulgaria' or 'North Macedonia'

cur.execute(f"""
    SELECT year, gap_percent, unemployment
    FROM wage_gap_practice
    WHERE country = '{country_name}'
    ORDER BY year DESC
""")
results = cur.fetchall()

print(f"Wage gap in {country_name}:")
for row in results:
    print(f"  {row[0]}: Gap = {row[1]}%, Unemployment = {row[2]}%")


# ============================================================
# EXERCISE 3: Calculate average wage gap by country
# ============================================================
print("\n\n📈 EXERCISE 3: Average wage gap comparison")
print("-" * 70)

cur.execute("""
    SELECT
        country,
        ROUND(AVG(gap_percent), 2) as avg_gap,
        ROUND(AVG(unemployment), 2) as avg_unemployment
    FROM wage_gap_practice
    GROUP BY country
    ORDER BY avg_gap DESC
""")

results = cur.fetchall()

print("Country Rankings by Average Wage Gap:")
rank = 1
for row in results:
    print(f"  {rank}. {row[0]:20} - Avg Gap: {row[1]}% | Avg Unemployment: {row[2]}%")
    rank += 1


# ============================================================
# EXERCISE 4: Insert your own data!
# ============================================================
print("\n\n✍️  EXERCISE 4: Add new data to the database")
print("-" * 70)

# 👉 TRY ADDING DATA FOR A NEW COUNTRY:
# Uncomment these lines to insert new data:

# new_country = 'Croatia'
# new_year = 2023
# new_gap = 11.5
# new_unemployment = 6.2
#
# cur.execute("""
#     INSERT INTO wage_gap_practice (country, year, gap_percent, unemployment)
#     VALUES (%s, %s, %s, %s)
# """, (new_country, new_year, new_gap, new_unemployment))
# conn.commit()  # IMPORTANT: Save changes to database!
# print(f"✅ Added {new_country} data to database!")


# ============================================================
# EXERCISE 5: Use pandas for analysis
# ============================================================
print("\n\n🐼 EXERCISE 5: Load into pandas and analyze")
print("-" * 70)

df = pd.read_sql_query("""
    SELECT
        w.country,
        w.year,
        w.gap_percent,
        w.unemployment,
        c.population,
        c.gdp_billions
    FROM wage_gap_practice w
    JOIN countries c ON w.country = c.name
    ORDER BY w.gap_percent DESC
""", conn)

print("\nCombined Dataset:")
print(df.to_string(index=False))

print("\n\n📊 Quick Statistics:")
print(f"  Highest wage gap: {df['gap_percent'].max()}% ({df.loc[df['gap_percent'].idxmax(), 'country']})")
print(f"  Lowest wage gap: {df['gap_percent'].min()}% ({df.loc[df['gap_percent'].idxmin(), 'country']})")
print(f"  Average across all: {df['gap_percent'].mean():.2f}%")


# ============================================================
# EXERCISE 6: YOUR CUSTOM QUERY
# ============================================================
print("\n\n🎯 EXERCISE 6: Write your own query!")
print("-" * 70)

# 👉 TRY WRITING YOUR OWN QUERY HERE:
# Examples:
# - Find countries with GDP > 50 billion
# - Get data for year 2023 only
# - Calculate correlation between unemployment and wage gap

# YOUR QUERY HERE:
# cur.execute("SELECT ... FROM ... WHERE ...")
# results = cur.fetchall()
# for row in results:
#     print(row)

print("✏️  Uncomment the code above and write your own SQL query!")


# Close connection
cur.close()
conn.close()

print("\n" + "=" * 70)
print("PRACTICE SESSION COMPLETE!")
print("=" * 70)
print("\n🎓 WHAT TO TRY NEXT:")
print("  1. Change the values in EXERCISE 1, 2, 3")
print("  2. Uncomment EXERCISE 4 to insert new data")
print("  3. Write your own query in EXERCISE 6")
print("  4. Run this script multiple times with different values")
print("\n💡 TIP: Each time you modify this file, save it and run again!")
print("=" * 70)
