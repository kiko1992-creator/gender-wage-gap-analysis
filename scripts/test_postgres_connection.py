"""
PostgreSQL Connection Tutorial
Learn how to connect Python to PostgreSQL database

Run this script: python scripts/test_postgres_connection.py
"""

import psycopg2
import pandas as pd

print("=" * 60)
print("POSTGRESQL + PYTHON CONNECTION TUTORIAL")
print("=" * 60)

# Step 1: Connect to PostgreSQL
print("\n📡 Step 1: Connecting to database...")
try:
    # Try Unix socket first (Linux/Mac)
    conn = psycopg2.connect(
        dbname="practice_db",
        user="postgres",
        host="/var/run/postgresql"
    )
    print("✅ Connection successful! (Unix socket)")
except:
    try:
        # Fallback to TCP/IP (Windows)
        conn = psycopg2.connect(
            dbname="practice_db",
            user="postgres",
            password="",  # Add password if needed
            host="localhost",
            port="5432"
        )
        print("✅ Connection successful! (TCP/IP - localhost)")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 If using Windows, you might need to:")
        print("   1. Add password in line 26 of this script")
        print("   2. Make sure PostgreSQL service is running")
        exit(1)

# Step 2: Create cursor (pointer to execute queries)
print("\n🔍 Step 2: Creating cursor...")
cur = conn.cursor()

# Step 3: Execute a simple query
print("\n📊 Step 3: Fetching countries data...")
cur.execute("SELECT * FROM countries ORDER BY population DESC;")
rows = cur.fetchall()

print("\nCountries in database:")
print("-" * 60)
for row in rows:
    print(f"ID: {row[0]} | Name: {row[1]:20} | Pop: {row[2]:,} | GDP: ${row[3]}B")

# Step 4: Load into pandas DataFrame
print("\n📈 Step 4: Loading into pandas DataFrame...")
df = pd.read_sql_query("SELECT * FROM countries;", conn)
print("\nAs pandas DataFrame:")
print(df)

# Step 5: More complex query - wage gap analysis
print("\n💰 Step 5: Wage gap analysis...")
query = """
    SELECT
        country,
        year,
        gap_percent,
        unemployment,
        CASE
            WHEN gap_percent > 12 THEN 'High Gap'
            WHEN gap_percent > 10 THEN 'Medium Gap'
            ELSE 'Low Gap'
        END as gap_category
    FROM wage_gap_practice
    ORDER BY gap_percent DESC;
"""

df_wage = pd.read_sql_query(query, conn)
print("\nWage Gap Analysis:")
print(df_wage)

# Step 6: Calculate statistics
print("\n📊 Step 6: Statistical summary...")
summary_query = """
    SELECT
        country,
        COUNT(*) as observations,
        ROUND(AVG(gap_percent), 2) as avg_gap,
        ROUND(MIN(gap_percent), 2) as min_gap,
        ROUND(MAX(gap_percent), 2) as max_gap
    FROM wage_gap_practice
    GROUP BY country
    ORDER BY avg_gap DESC;
"""

df_summary = pd.read_sql_query(summary_query, conn)
print("\nSummary by Country:")
print(df_summary.to_string(index=False))

# Step 7: Close connection (important!)
print("\n🔒 Step 7: Closing connection...")
cur.close()
conn.close()
print("✅ Connection closed safely")

print("\n" + "=" * 60)
print("TUTORIAL COMPLETE! 🎉")
print("=" * 60)
print("\nWhat you learned:")
print("✅ Connect Python to PostgreSQL")
print("✅ Execute SQL queries from Python")
print("✅ Load data into pandas DataFrame")
print("✅ Perform calculations in SQL")
print("✅ Properly close connections")
print("\nNext steps:")
print("1. Try modifying the queries")
print("2. Insert your own data")
print("3. Create new tables")
print("=" * 60)
