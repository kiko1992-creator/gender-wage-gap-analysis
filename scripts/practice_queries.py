"""
INTERACTIVE PRACTICE: Modify this script!

Try changing the queries below and see what happens.
Run: sudo python scripts/practice_queries.py
"""

import psycopg2

print("\n🎯 INTERACTIVE POSTGRESQL PRACTICE")
print("=" * 60)

# 1. CONNECT (Don't change this part)
conn = psycopg2.connect(dbname="practice_db", user="postgres", host="/var/run/postgresql")
cur = conn.cursor()
print("✅ Connected to database!\n")

# ============================================================
# EXERCISE 1: Get all countries
# ============================================================
print("📊 EXERCISE 1: Get all countries")
print("-" * 60)

cur.execute("SELECT * FROM countries;")
results = cur.fetchall()

for row in results:
    print(f"  {row[1]:20} - Population: {row[2]:,}")

# ============================================================
# EXERCISE 2: Get only large countries (population > 3 million)
# ============================================================
print("\n📊 EXERCISE 2: Large countries only")
print("-" * 60)

# TRY CHANGING THE NUMBER BELOW (try 5000000, 2000000, etc.)
cur.execute("SELECT * FROM countries WHERE population > 3000000;")
results = cur.fetchall()

print(f"Found {len(results)} countries with population > 3 million:")
for row in results:
    print(f"  ✓ {row[1]:20} - {row[2]:,} people")

# ============================================================
# EXERCISE 3: Get wage gap data for specific country
# ============================================================
print("\n📊 EXERCISE 3: Wage gap for specific country")
print("-" * 60)

# TRY CHANGING 'Serbia' to 'North Macedonia' or 'Bulgaria'
country_name = 'Serbia'

cur.execute(f"""
    SELECT year, gap_percent, unemployment
    FROM wage_gap_practice
    WHERE country = '{country_name}'
    ORDER BY year DESC;
""")
results = cur.fetchall()

print(f"Wage gap data for {country_name}:")
for row in results:
    print(f"  Year {row[0]}: Gap = {row[1]}%, Unemployment = {row[2]}%")

# ============================================================
# EXERCISE 4: Calculate statistics
# ============================================================
print("\n📊 EXERCISE 4: Calculate averages")
print("-" * 60)

cur.execute("""
    SELECT
        country,
        COUNT(*) as years_of_data,
        ROUND(AVG(gap_percent), 2) as avg_gap,
        ROUND(AVG(unemployment), 2) as avg_unemployment
    FROM wage_gap_practice
    GROUP BY country
    ORDER BY avg_gap DESC;
""")
results = cur.fetchall()

print("Average statistics by country:")
print(f"  {'Country':<20} {'Years':>8} {'Avg Gap':>10} {'Avg Unemp':>12}")
print("  " + "-" * 55)
for row in results:
    print(f"  {row[0]:<20} {row[1]:>8} {row[2]:>9}% {row[3]:>11}%")

# ============================================================
# YOUR TURN: Write your own query!
# ============================================================
print("\n📊 YOUR TURN: Try your own query")
print("-" * 60)
print("Uncomment the lines below and write your own SQL query:\n")

# YOUR_QUERY = """
#     SELECT * FROM countries WHERE name = 'Bulgaria';
# """
# cur.execute(YOUR_QUERY)
# results = cur.fetchall()
# print(results)

# ============================================================
# CLOSE (Don't change this part)
# ============================================================
cur.close()
conn.close()
print("\n✅ Connection closed")

# ============================================================
# CHALLENGES: Try to figure out these queries
# ============================================================
print("\n" + "=" * 60)
print("🎯 CHALLENGES (Try to write these queries):")
print("=" * 60)
print("""
1. Get countries ordered by GDP (highest first)
   Hint: SELECT * FROM countries ORDER BY gdp_billions DESC;

2. Find countries with gap > 10%
   Hint: SELECT * FROM wage_gap_practice WHERE gap_percent > 10;

3. Count total rows in wage_gap_practice
   Hint: SELECT COUNT(*) FROM wage_gap_practice;

4. Get only 2023 data
   Hint: SELECT * FROM wage_gap_practice WHERE year = 2023;

5. Find country with highest single wage gap value
   Hint: SELECT country, MAX(gap_percent) FROM wage_gap_practice GROUP BY country;
""")
print("=" * 60)
