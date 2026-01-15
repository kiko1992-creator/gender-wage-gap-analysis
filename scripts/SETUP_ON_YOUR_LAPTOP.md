# 💻 How to Connect YOUR Laptop to This PostgreSQL Tutorial

**Goal:** Get the PostgreSQL tutorials working on your laptop in VS Code

**Time:** 10 minutes

---

## 🎯 **Step-by-Step Setup**

### **STEP 1: Open VS Code** ✅

1. Open Visual Studio Code
2. Open your project folder: `gender-wage-gap-analysis`

---

### **STEP 2: Open Terminal in VS Code** (30 seconds)

1. Look at the **top menu** bar
2. Click **"Terminal"**
3. Click **"New Terminal"**
4. A dark box appears at the bottom ✅

---

### **STEP 3: Install Python PostgreSQL Library** (1 minute)

In the terminal at the bottom of VS Code, type:

```bash
pip install psycopg2-binary pandas
```

**You should see:**
```
Successfully installed psycopg2-binary-2.9.x pandas-2.x.x
```

✅ **Done!** Python can now talk to PostgreSQL.

---

### **STEP 4: Run the Database Setup Script** (2 minutes)

This creates the practice database and adds sample data.

**In VS Code terminal, type:**

```bash
python scripts/setup_database_windows.py
```

**You should see:**
```
============================================================
POSTGRESQL DATABASE SETUP
============================================================

📡 Connecting to PostgreSQL...
✅ Connected to PostgreSQL!

🗄️  Creating practice_db database...
✅ Database 'practice_db' created!

📊 Creating tables...
  ✅ Countries table created!
  ✅ Wage gap table created!

📥 Inserting sample data...
✅ Sample data inserted!

✅ SETUP COMPLETE!
```

✅ **Success!** Your database is ready.

---

### **STEP 5: Run Your First Tutorial** (2 minutes)

```bash
python scripts/test_postgres_connection.py
```

**You'll see a LOT of output!** This is GOOD!

The script will show you:
- ✅ How Python connects to PostgreSQL
- ✅ Your 3 countries (Serbia, Bulgaria, North Macedonia)
- ✅ Wage gap analysis
- ✅ Statistical calculations

**Read through the output** - it explains each step!

---

### **STEP 6: Start Practicing!** (10+ minutes)

```bash
python scripts/practice_queries_interactive.py
```

This runs exercises you can modify!

**Then:**
1. Open `scripts/practice_queries_interactive.py` in VS Code
2. Change line 39: `min_population = 2000000` (see all countries)
3. Save the file (Ctrl+S or Cmd+S)
4. Run again: `python scripts/practice_queries_interactive.py`
5. See the different results! 🎉

---

## 🚨 **If You Get Errors:**

### **Error 1: "psycopg2" not found**
**Solution:** Run `pip install psycopg2-binary` again

### **Error 2: "connection to server failed"**
**Solution:** PostgreSQL isn't running.

**Windows:**
1. Press `Windows Key + R`
2. Type: `services.msc` and press Enter
3. Find "postgresql" in the list
4. Right-click → "Start"

**Mac:**
```bash
brew services start postgresql
```

### **Error 3: "password authentication failed"**
**Solution:** You set a password during PostgreSQL installation.

**Edit the setup script:**
1. Open `scripts/setup_database_windows.py` in VS Code
2. Line 18: Change `password=""` to `password="YOUR_PASSWORD"`
3. Save and run again

### **Error 4: "database already exists"**
**Solution:** That's fine! Skip to Step 5 (run the tutorial).

---

## ✅ **Quick Test - Did It Work?**

After Step 5, you should have seen:

```
Countries in database:
------------------------------------------------------------
ID: 2 | Name: Serbia               | Pop: 6,899,000 | GDP: $63.10B
ID: 3 | Name: Bulgaria             | Pop: 6,877,000 | GDP: $84.10B
ID: 1 | Name: North Macedonia      | Pop: 2,083,000 | GDP: $13.80B
```

**If you see this → YOU'RE SET UP!** ✅

---

## 🎓 **What You Now Have:**

✅ PostgreSQL running on your laptop
✅ Practice database with sample data
✅ Python connected to PostgreSQL
✅ 6 tutorial files ready to use
✅ Interactive scripts you can modify

---

## 📚 **What to Do Next:**

1. **Read:** Open `scripts/QUICK_START.md` in VS Code
2. **Practice:** Modify `scripts/practice_queries_interactive.py`
3. **Learn:** Open `scripts/VISUAL_EXPLANATION.md` for diagrams

---

## 🎯 **Your Learning Path:**

```
✅ Day 1: Set up PostgreSQL on laptop (YOU'RE HERE!)
👉 Day 2: Practice modifying queries
   Day 3: Write custom SQL queries
   Day 4: Learn JOINs
   Week 2: Connect to Eurostat API
   Week 3: Build automated pipeline
   Week 4: Scale to 27 EU countries
```

---

## 💡 **What These Files Do:**

| File | Purpose |
|------|---------|
| `setup_database_windows.py` | Creates database (run ONCE) |
| `test_postgres_connection.py` | Complete tutorial |
| `practice_queries_interactive.py` | Hands-on exercises |
| `QUICK_START.md` | Fast reference guide |
| `START_HERE.md` | Beginner's guide |
| `VISUAL_EXPLANATION.md` | Diagrams & analogies |

---

## 🚀 **Ready to Start?**

```bash
# 1. Install library
pip install psycopg2-binary pandas

# 2. Setup database
python scripts/setup_database_windows.py

# 3. Run tutorial
python scripts/test_postgres_connection.py

# 4. Practice!
python scripts/practice_queries_interactive.py
```

**You've got this!** 💪

---

## ❓ **Still Stuck?**

Check which step failed and look at the troubleshooting section above.

Common issues:
- PostgreSQL service not running → Start it in services.msc
- Password needed → Add password to connection scripts
- Port 5432 already used → Another app is using PostgreSQL's port

Tell me what error you see and I'll help you fix it! 🔧
