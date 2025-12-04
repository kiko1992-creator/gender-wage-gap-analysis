# Debugging Guide & Potential Issues

## Quick Start Testing

**Before running the main analysis, run the test notebook:**

```bash
jupyter notebook notebooks/00_test_and_debug.ipynb
```

This will verify all libraries work and data loads correctly.

---

## Potential Issues & Solutions

### 1. **Missing Dependencies**

**Symptom:** `ModuleNotFoundError` or `ImportError`

**Solution:**
```bash
pip install -r requirements.txt
```

**Critical libraries:**
- pandas >= 2.3.3
- numpy >= 2.3.5
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- scipy >= 1.14.0
- scikit-learn >= 1.3.0 (ADDED - needed for regression)

---

### 2. **Matplotlib Style Issues**

**Symptom:** `OSError: Matplotlib style 'seaborn-v0_8-whitegrid' not found`

**Fix:** In notebooks, change:
```python
# FROM:
plt.style.use('seaborn-v0_8-whitegrid')

# TO:
plt.style.use('seaborn-whitegrid')
# OR
plt.style.use('default')
```

**Location:** Cell 2 in `05_comprehensive_analysis.ipynb`

---

### 3. **File Path Issues**

**Symptom:** `FileNotFoundError` when loading data

**Check:**
```python
import os
print(os.getcwd())  # Should show gender-wage-gap-analysis directory
```

**Fix:**
- Run notebooks from project root OR
- Adjust paths in notebooks:
```python
# If running from notebooks/ directory:
df = pd.read_csv('../data/raw/expanded_balkan_wage_data.csv')

# If running from project root:
df = pd.read_csv('data/raw/expanded_balkan_wage_data.csv')
```

---

### 4. **Small Sample Size Warnings**

**Symptom:** Warnings like "Sample size too small for accurate results"

**This is EXPECTED** for some subgroups (Kosovo, Montenegro, Bosnia)

**Not an error** - just informational. The analysis handles this appropriately.

---

### 5. **Statistical Test Errors**

**Symptom:** Error in t-test or regression with insufficient data

**Cause:** Some country-year combinations have only 1-2 data points

**Already handled** in code with:
```python
if len(data) > 2:
    # Run test
else:
    # Skip or use alternative
```

---

### 6. **Visualization Display Issues**

**Symptom:** Plots don't display in Jupyter

**Solutions:**
1. Add at start of notebook:
```python
%matplotlib inline
```

2. Or use:
```python
%matplotlib widget  # For interactive plots
```

3. Check backend:
```python
import matplotlib
print(matplotlib.get_backend())
```

---

### 7. **Memory Issues (Unlikely)**

**Symptom:** Kernel crashes or "Out of Memory"

**Solution:** Dataset is small (~100 records), so this shouldn't happen
- If it does, restart kernel: `Kernel > Restart`
- Close other applications

---

### 8. **Unicode/Encoding Issues**

**Symptom:** Strange characters in country names

**Already handled** in CSV with UTF-8 encoding

**If issues persist:**
```python
df = pd.read_csv('data.csv', encoding='utf-8-sig')
```

---

### 9. **Jupyter Kernel Issues**

**Symptom:** "Kernel not found" or "No kernel"

**Solution:**
```bash
python -m ipykernel install --user --name=gender-analysis
```

Then select kernel in Jupyter: `Kernel > Change Kernel > gender-analysis`

---

## Known Issues & Workarounds

### Issue #1: Plot Style in Seaborn

**Status:** FIXED in test notebook
**Workaround:** Use `sns.set_style('whitegrid')` instead of matplotlib style

### Issue #2: Linear Regression with Few Points

**Status:** HANDLED
**Code checks:** `if len(yearly) > 2:` before running regression

### Issue #3: Missing Data in Some Countries

**Status:** EXPECTED
**Note:** Kosovo, Bosnia, Montenegro have limited data (2-4 records each)

---

## Validation Checklist

Before running full analysis:

- [ ] All libraries installed (`pip list | grep pandas`)
- [ ] Data files exist in `data/raw/`
- [ ] Test notebook runs without errors
- [ ] Simple plot displays correctly
- [ ] Can load and filter data

Run validation script:
```bash
python scripts/validate_data.py
```

---

## Debugging Workflow

1. **Run Test Notebook First**
   - `notebooks/00_test_and_debug.ipynb`
   - Identifies issues immediately

2. **Check Each Section**
   - If error occurs, note the cell number
   - Read error message carefully
   - Check if it's a warning (orange) or error (red)

3. **Common Error Patterns**

   **Pattern 1: Import Error**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Fix:** `pip install scikit-learn`

   **Pattern 2: Key Error**
   ```
   KeyError: 'Female'
   ```
   **Cause:** Data filtering resulted in empty subset
   **Check:** Are you filtering correctly?

   **Pattern 3: Deprecation Warnings**
   ```
   FutureWarning: ...
   ```
   **Action:** These are just warnings, analysis still works

4. **Restart Fresh**
   ```python
   # In Jupyter:
   # Kernel > Restart & Clear Output
   # Then: Cell > Run All
   ```

---

## Performance Notes

**Expected Runtime:**
- Test notebook: ~30 seconds
- Full analysis (05_comprehensive_analysis.ipynb): ~2-3 minutes
- All notebooks: ~10 minutes

**If slower:**
- Close other applications
- Check CPU usage
- Restart Jupyter

---

## Getting Help

**Error Log:**
When reporting issues, include:
1. Full error message (copy from notebook)
2. Cell number where error occurred
3. Python version: `python --version`
4. Library versions: `pip list`

**Quick Diagnostics:**
```python
import sys
import pandas as pd
import numpy as np

print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
```

---

## Success Indicators

**✓ Everything Working If:**
1. Test notebook completes without errors
2. Plots display correctly
3. Statistical tests return p-values
4. No "CRITICAL ERROR" messages
5. Visualizations appear

**⚠ Acceptable Warnings:**
- "Sample size small" for some countries
- FutureWarning about pandas
- Matplotlib style warnings (if using old version)

**✗ Must Fix:**
- ModuleNotFoundError
- FileNotFoundError
- SyntaxError
- Kernel crashes

---

## Advanced Debugging

### Enable Verbose Output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data Quality:
```python
df.info()
df.describe()
df.isnull().sum()
df['column'].value_counts()
```

### Test Statistical Functions:
```python
from scipy import stats
# Test with simple data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
result = stats.ttest_ind(x, y)
print(result)  # Should work
```

---

## Environment Setup (Fresh Install)

If starting from scratch:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Jupyter kernel
python -m ipykernel install --user --name=gender-analysis

# 5. Start Jupyter
jupyter notebook

# 6. Open 00_test_and_debug.ipynb first
```

---

## Final Checklist Before Running Full Analysis

✓ Requirements installed
✓ Data files present
✓ Test notebook passes
✓ Can import all libraries
✓ Simple plot displays
✓ Statistical functions work
✓ No critical errors in test

**If all checked, proceed to:**
`notebooks/05_comprehensive_analysis.ipynb`

---

**Last Updated:** December 2025
**Tested With:** Python 3.11+, Pandas 2.3+, NumPy 2.3+
