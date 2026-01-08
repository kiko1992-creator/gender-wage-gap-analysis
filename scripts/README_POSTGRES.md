# PostgreSQL Quick Reference Guide

## What You Have Set Up

### Database: `practice_db`

**Tables:**
1. `countries` - Sample country data
2. `wage_gap_practice` - Sample wage gap data

### How to Access

**From Terminal (SQL):**
```bash
sudo -u postgres psql -d practice_db
```

**From Python:**
```python
python scripts/test_postgres_connection.py
```

---

## Common SQL Commands

### View all tables:
```sql
\dt
```

### View table structure:
```sql
\d countries
```

### Query data:
```sql
SELECT * FROM countries;
SELECT * FROM wage_gap_practice ORDER BY gap_percent DESC;
```

### Calculate averages:
```sql
SELECT country, AVG(gap_percent) as avg_gap
FROM wage_gap_practice
GROUP BY country;
```

### Insert new data:
```sql
INSERT INTO wage_gap_practice (country, year, gap_percent, unemployment)
VALUES ('Romania', 2023, 11.2, 5.6);
```

### Update data:
```sql
UPDATE wage_gap_practice
SET gap_percent = 15.5
WHERE country = 'North Macedonia' AND year = 2023;
```

### Delete data:
```sql
DELETE FROM wage_gap_practice WHERE year < 2020;
```

---

## Next Steps (Week 1 Learning Path)

### ✅ Day 1-2: COMPLETED
- [x] Installed PostgreSQL
- [x] Created first database
- [x] Created sample tables
- [x] Inserted data
- [x] Ran queries

### 📝 Day 3-4: Practice Exercises

**Exercise 1:** Create a new table for unemployment data
```sql
CREATE TABLE unemployment_data (
    id SERIAL PRIMARY KEY,
    country VARCHAR(100),
    year INTEGER,
    rate DECIMAL(5,2)
);
```

**Exercise 2:** Join tables to combine data
```sql
SELECT c.name, w.year, w.gap_percent
FROM countries c
JOIN wage_gap_practice w ON c.name = w.country;
```

**Exercise 3:** Calculate statistics
```sql
-- Find countries with gap > 10%
SELECT * FROM wage_gap_practice WHERE gap_percent > 10;

-- Count observations per country
SELECT country, COUNT(*) FROM wage_gap_practice GROUP BY country;
```

### 📅 Day 5-7: Python Integration

Run the tutorial script:
```bash
sudo python scripts/test_postgres_connection.py
```

Modify the script to:
1. Insert new data from Python
2. Update existing records
3. Create your own queries

---

## Troubleshooting

**PostgreSQL not running:**
```bash
sudo service postgresql start
```

**Can't connect:**
```bash
# Check if service is running
sudo service postgresql status
```

**Permission denied:**
```bash
# Run as postgres user
sudo -u postgres psql
```

---

## Resources

- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [SQL Practice](https://sqlbolt.com/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)

---

## Your Progress

- [x] Week 1 Day 1-2: PostgreSQL setup ✅
- [ ] Week 1 Day 3-4: Practice exercises
- [ ] Week 1 Day 5-7: Python integration
- [ ] Week 2: API Integration
- [ ] Week 3: Build pipeline
- [ ] Week 4: Scale to project
