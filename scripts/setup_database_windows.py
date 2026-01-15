"""
Setup PostgreSQL Database on Windows/Mac
This creates the practice database and tables for the tutorial

Run this ONCE: python scripts/setup_database_windows.py
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

print("=" * 70)
print("POSTGRESQL DATABASE SETUP")
print("=" * 70)

# Step 1: Connect to default postgres database
print("\n📡 Connecting to PostgreSQL...")
try:
    # Try common connection methods
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="",  # Leave empty if no password set
        host="localhost",
        port="5432"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    print("✅ Connected to PostgreSQL!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n💡 TIP: You might need to provide a password.")
    print("   Edit this file (line 18-19) and add your PostgreSQL password.")
    print("   Or check if PostgreSQL service is running in services.msc")
    exit(1)

cur = conn.cursor()

# Step 2: Create practice_db database
print("\n🗄️  Creating practice_db database...")
try:
    cur.execute("CREATE DATABASE practice_db;")
    print("✅ Database 'practice_db' created!")
except Exception as e:
    if "already exists" in str(e):
        print("ℹ️  Database 'practice_db' already exists - that's fine!")
    else:
        print(f"❌ Error: {e}")

cur.close()
conn.close()

# Step 3: Connect to practice_db and create tables
print("\n📊 Creating tables...")
conn = psycopg2.connect(
    dbname="practice_db",
    user="postgres",
    password="",  # Same as above
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Create countries table
print("  Creating 'countries' table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS countries (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        population INTEGER,
        gdp_billions DECIMAL(10,2)
    );
""")
print("  ✅ Countries table created!")

# Create wage_gap_practice table
print("  Creating 'wage_gap_practice' table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS wage_gap_practice (
        id SERIAL PRIMARY KEY,
        country VARCHAR(100),
        year INTEGER,
        gap_percent DECIMAL(5,2),
        unemployment DECIMAL(5,2)
    );
""")
print("  ✅ Wage gap table created!")

# Insert sample data into countries
print("\n📥 Inserting sample data...")
cur.execute("""
    INSERT INTO countries (name, population, gdp_billions) VALUES
    ('North Macedonia', 2083000, 13.8),
    ('Serbia', 6899000, 63.1),
    ('Bulgaria', 6877000, 84.1)
    ON CONFLICT DO NOTHING;
""")

# Insert sample data into wage_gap_practice
cur.execute("""
    INSERT INTO wage_gap_practice (country, year, gap_percent, unemployment) VALUES
    ('North Macedonia', 2023, 15.2, 13.5),
    ('North Macedonia', 2022, 14.8, 14.2),
    ('Serbia', 2023, 9.3, 9.8),
    ('Serbia', 2022, 9.7, 10.1),
    ('Bulgaria', 2023, 12.4, 4.3)
    ON CONFLICT DO NOTHING;
""")

conn.commit()
print("✅ Sample data inserted!")

# Verify setup
print("\n✅ Verifying setup...")
cur.execute("SELECT COUNT(*) FROM countries;")
country_count = cur.fetchone()[0]
print(f"  Countries table: {country_count} records")

cur.execute("SELECT COUNT(*) FROM wage_gap_practice;")
wage_count = cur.fetchone()[0]
print(f"  Wage gap table: {wage_count} records")

cur.close()
conn.close()

print("\n" + "=" * 70)
print("✅ SETUP COMPLETE!")
print("=" * 70)
print("\nYour database is ready! Next steps:")
print("1. Run the tutorial: python scripts/test_postgres_connection.py")
print("2. Practice queries: python scripts/practice_queries_interactive.py")
print("3. Read guides: Open scripts/QUICK_START.md")
print("=" * 70)
