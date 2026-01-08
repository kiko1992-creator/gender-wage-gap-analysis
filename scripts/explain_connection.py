"""
POSTGRESQL + PYTHON CONNECTION EXPLAINED
=========================================

This script shows EXACTLY what happens at each step
with detailed annotations and print statements.

Run: sudo python scripts/explain_connection.py
"""

import psycopg2

print("\n" + "="*70)
print("STEP-BY-STEP: HOW PYTHON TALKS TO POSTGRESQL")
print("="*70)

# ============================================================================
# STEP 1: ESTABLISH CONNECTION (Like dialing a phone number)
# ============================================================================
print("\n📞 STEP 1: ESTABLISHING CONNECTION")
print("-" * 70)

print("\nWhat we're doing:")
print("  - Telling Python: 'I want to talk to the practice_db database'")
print("  - PostgreSQL is like a librarian waiting for requests")
print("  - The connection is like a phone line between Python and PostgreSQL")

print("\nCode:")
print("  conn = psycopg2.connect(dbname='practice_db', user='postgres', host='/var/run/postgresql')")

try:
    conn = psycopg2.connect(
        dbname="practice_db",      # Which database? (like calling specific library)
        user="postgres",            # Who are you? (like showing ID card)
        host="/var/run/postgresql"  # Where is it? (like phone number)
    )
    print("\n✅ RESULT: Connection established!")
    print(f"   Connection object created: {conn}")
    print(f"   Type: {type(conn)}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    exit(1)

# ============================================================================
# STEP 2: CREATE CURSOR (Like having a conversation)
# ============================================================================
print("\n\n🗣️ STEP 2: CREATING CURSOR")
print("-" * 70)

print("\nWhat we're doing:")
print("  - Cursor = your voice in the conversation")
print("  - It carries your questions to the database")
print("  - It brings back the answers")

print("\nCode:")
print("  cur = conn.cursor()")

cur = conn.cursor()
print("\n✅ RESULT: Cursor created!")
print(f"   Cursor object: {cur}")
print(f"   Status: Ready to send queries")

# ============================================================================
# STEP 3: EXECUTE QUERY (Ask a question)
# ============================================================================
print("\n\n❓ STEP 3: EXECUTING SQL QUERY")
print("-" * 70)

print("\nWhat we're doing:")
print("  - We're asking: 'Show me all countries'")
print("  - PostgreSQL searches through the 'countries' table")
print("  - It prepares the results to send back")

query = "SELECT * FROM countries;"
print(f"\nSQL Query:")
print(f"  {query}")

print("\nWhat this means in English:")
print("  SELECT * = 'Give me all columns'")
print("  FROM countries = 'from the countries table'")
print("  ; = 'end of query'")

cur.execute(query)
print("\n✅ RESULT: Query executed!")
print("   PostgreSQL has found the data and is ready to send it")

# ============================================================================
# STEP 4: FETCH RESULTS (Listen to the answer)
# ============================================================================
print("\n\n📥 STEP 4: FETCHING RESULTS")
print("-" * 70)

print("\nWhat we're doing:")
print("  - We're listening to PostgreSQL's response")
print("  - It sends back the rows it found")
print("  - Python receives them as a list of tuples")

rows = cur.fetchall()

print(f"\n✅ RESULT: Received {len(rows)} rows")
print("\nData structure:")
print(f"  Type: {type(rows)}")
print(f"  It's a list: {isinstance(rows, list)}")
print(f"  Each row is a tuple: {type(rows[0]) if rows else 'N/A'}")

print("\nRaw data:")
for i, row in enumerate(rows, 1):
    print(f"  Row {i}: {row}")

print("\nPretty format:")
print("  " + "-" * 65)
print(f"  {'ID':<5} {'Name':<20} {'Population':<15} {'GDP (B)':<10}")
print("  " + "-" * 65)
for row in rows:
    print(f"  {row[0]:<5} {row[1]:<20} {row[2]:<15,} ${row[3]:<10}")
print("  " + "-" * 65)

# ============================================================================
# STEP 5: MORE COMPLEX QUERY (Filtering and calculating)
# ============================================================================
print("\n\n🔍 STEP 5: ADVANCED QUERY (Filter & Calculate)")
print("-" * 70)

print("\nWhat we're doing:")
print("  - Ask for Serbian data only")
print("  - Calculate average wage gap")
print("  - PostgreSQL does the math for us!")

query2 = """
    SELECT
        country,
        COUNT(*) as total_years,
        AVG(gap_percent) as avg_gap,
        MIN(gap_percent) as min_gap,
        MAX(gap_percent) as max_gap
    FROM wage_gap_practice
    WHERE country = 'Serbia'
    GROUP BY country;
"""

print("\nSQL Query:")
for line in query2.strip().split('\n'):
    print(f"  {line}")

print("\nWhat this means in English:")
print("  SELECT country = 'Show me the country name'")
print("  COUNT(*) = 'Count how many rows'")
print("  AVG(gap_percent) = 'Calculate average gap'")
print("  MIN/MAX = 'Find smallest/largest gap'")
print("  FROM wage_gap_practice = 'from wage gap table'")
print("  WHERE country = 'Serbia' = 'only Serbian data'")
print("  GROUP BY country = 'organize by country'")

cur.execute(query2)
result = cur.fetchone()  # Get one row

print("\n✅ RESULT:")
if result:
    print(f"  Country: {result[0]}")
    print(f"  Total years of data: {result[1]}")
    print(f"  Average gap: {result[2]:.2f}%")
    print(f"  Minimum gap: {result[3]:.2f}%")
    print(f"  Maximum gap: {result[4]:.2f}%")

    print("\n💡 INSIGHT:")
    print(f"  PostgreSQL searched through all wage_gap_practice rows,")
    print(f"  filtered for Serbia, and calculated 4 statistics")
    print(f"  All in < 0.001 seconds!")

# ============================================================================
# STEP 6: CLOSE CONNECTION (Hang up the phone)
# ============================================================================
print("\n\n🔒 STEP 6: CLOSING CONNECTION")
print("-" * 70)

print("\nWhat we're doing:")
print("  - We're done with our questions")
print("  - Close the cursor (stop talking)")
print("  - Close the connection (hang up phone)")
print("  - This frees up resources")

cur.close()
conn.close()

print("\n✅ RESULT: Connection closed safely")
print("   Cursor closed: ✓")
print("   Connection closed: ✓")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("SUMMARY: THE COMPLETE CONVERSATION")
print("="*70)

print("""
1. CONNECT:    Python calls PostgreSQL → "Hello, can I access practice_db?"
2. CURSOR:     Create a channel → "I'm ready to ask questions"
3. EXECUTE:    Send SQL query → "Give me all countries"
4. FETCH:      Receive results → PostgreSQL sends back: [(1,'North Macedonia'...), ...]
5. QUERY AGAIN: Send another query → "Calculate Serbian average gap"
6. CLOSE:      End conversation → "Thanks, goodbye!"

ANALOGY:
--------
PostgreSQL = Librarian with organized filing system
Python = You, asking questions
Connection = Phone line between you and librarian
Cursor = Your voice carrying questions and answers
SQL Query = Your specific question in librarian's language
Results = Librarian's response with exact data you need

WHY THIS IS POWERFUL:
---------------------
CSV Method:
  - You manually search through all 146 rows
  - You write pandas code to filter
  - You calculate averages yourself
  - Takes seconds for large files

PostgreSQL Method:
  - Database searches through millions of rows instantly
  - Database does calculations
  - You just ask questions in SQL
  - Takes milliseconds even for huge datasets

NEXT STEPS:
-----------
1. Try modifying the queries above
2. Practice writing your own SELECT statements
3. Learn JOINs to combine tables
4. Then we'll fetch real data from Eurostat API and store it here!
""")

print("="*70)
print("END OF TUTORIAL")
print("="*70 + "\n")
