# 🚀 QUICK START - PostgreSQL Practice

## What You Have RIGHT NOW:

✅ PostgreSQL installed and running
✅ Practice database with sample data (3 countries)
✅ 2 tutorial scripts ready to use
✅ Connection between Python and PostgreSQL working!

---

## 3 Things You Can Do IMMEDIATELY:

### 1️⃣ **Run the Tutorial** (2 minutes)
```bash
sudo python scripts/test_postgres_connection.py
```
**What it shows:** Complete walkthrough of connecting Python to PostgreSQL

---

### 2️⃣ **Practice Modifying Queries** (10 minutes)
```bash
# Open this file in VS Code:
scripts/practice_queries_interactive.py

# Then modify the queries and run:
sudo python scripts/practice_queries_interactive.py
```

**Try These Modifications:**

**Change population filter:**
```python
# Line 31 - Try different values:
min_population = 2000000  # See all countries
min_population = 7000000  # See only largest countries
```

**Change country to analyze:**
```python
# Line 47 - Try different countries:
country_name = 'Bulgaria'
country_name = 'North Macedonia'
```

**Add your own data:**
```python
# Lines 92-102 - Uncomment these lines:
new_country = 'Croatia'
new_year = 2023
new_gap = 11.5
new_unemployment = 6.2
```

---

### 3️⃣ **Check Your Data** (Direct PostgreSQL Access)
```bash
# View all countries:
sudo -u postgres psql -d practice_db -c "SELECT * FROM countries;"

# View wage gap data:
sudo -u postgres psql -d practice_db -c "SELECT * FROM wage_gap_practice;"

# Calculate statistics:
sudo -u postgres psql -d practice_db -c "SELECT country, AVG(gap_percent) FROM wage_gap_practice GROUP BY country;"
```

---

## 📊 What's in Your Database:

**Table 1: countries**
- North Macedonia (2.08M pop, $13.8B GDP)
- Serbia (6.89M pop, $63.1B GDP)
- Bulgaria (6.87M pop, $84.1B GDP)

**Table 2: wage_gap_practice**
- 5 records spanning 2022-2023
- Includes: wage gap %, unemployment %
- Countries: North Macedonia, Serbia, Bulgaria

---

## 🎯 Learning Path (Next 7 Days):

**Day 1 (TODAY):** ✅ Run both scripts, understand connections
**Day 2:** Modify queries, insert new data
**Day 3:** Write custom SQL queries
**Day 4:** Learn JOINs (combine tables)
**Day 5:** Connect to Eurostat API
**Day 6:** Build mini-pipeline (API → PostgreSQL)
**Day 7:** Analyze real EU wage gap data

---

## 💡 Key Concepts You're Learning:

1. **Connection** = Highway between Python and PostgreSQL
2. **Cursor** = Your messenger that carries SQL queries
3. **Execute** = Send SQL command
4. **Fetch** = Get results back
5. **Commit** = Save changes permanently
6. **Close** = Clean up when done

---

## 🆘 Quick Troubleshooting:

**"Connection failed"**
→ Run: `sudo service postgresql start`

**"Permission denied"**
→ Use `sudo` before python commands

**"Database does not exist"**
→ Database already exists! Continue to next step.

---

## 📚 Your Files:

📄 `test_postgres_connection.py` - Complete tutorial walkthrough
📄 `practice_queries_interactive.py` - Hands-on practice with comments
📄 `START_HERE.md` - Beginner's guide
📄 `VISUAL_EXPLANATION.md` - Diagrams and analogies
📄 `README_POSTGRES.md` - Command reference

---

## 🎓 What You'll Build:

```
Week 1: Learn PostgreSQL basics ← YOU ARE HERE
   ↓
Week 2: Connect to real APIs (Eurostat, World Bank)
   ↓
Week 3: Build automated pipeline
   ↓
Week 4: Scale to 27 EU countries
   ↓
PhD: Analyze gender wage gap across EU + Your findings
```

---

## 🚀 START NOW:

```bash
# 1. Run tutorial
sudo python scripts/test_postgres_connection.py

# 2. Open interactive practice
code scripts/practice_queries_interactive.py

# 3. Modify and run
sudo python scripts/practice_queries_interactive.py
```

**You're ready! Go!** 💪
