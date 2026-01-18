# Visual Studio Code Setup Guide

## Quick Setup for This Project

### Step 1: Clone Repository to Your Local Machine

```bash
# Open terminal and navigate to where you want the project
cd ~/Documents  # or your preferred location

# Clone the repository
git clone https://github.com/kiko1992-creator/gender-wage-gap-analysis.git

# Navigate into project
cd gender-wage-gap-analysis

# Checkout the working branch
git checkout claude/streamlit-production-optimization-BvCxo
```

### Step 2: Install VS Code Extensions

Open VS Code and install these essential extensions:

1. **Python** (Microsoft) - Required
2. **Pylance** (Microsoft) - Required for IntelliSense
3. **Jupyter** (Microsoft) - For notebooks
4. **PostgreSQL** (Chris Kolkman) - Database management
5. **GitLens** (GitKraken) - Advanced git features
6. **Python Indent** (Kevin Rose) - Better auto-indentation
7. **autoDocstring** (Nils Werner) - Generate docstrings
8. **Error Lens** - Inline error messages

**Optional but recommended**:
- **GitHub Copilot** or **Continue** (for AI assistance in VS Code)
- **Prettier** - Code formatting
- **TODO Highlight** - Track TODOs
- **Material Icon Theme** - Better file icons

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 mypy ipython jupyter
```

### Step 4: Configure VS Code Settings

Create/edit `.vscode/settings.json` in your project root:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/*.pyc": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "[python]": {
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### Step 5: Set Up PostgreSQL Connection (Local)

**On Linux (Ubuntu/Debian)**:
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres createdb practice_db

# Set password (optional for local dev)
sudo -u postgres psql
ALTER USER postgres PASSWORD 'your_password';
\q
```

**On Mac**:
```bash
# Install via Homebrew
brew install postgresql@16
brew services start postgresql@16

# Create database
createdb practice_db
```

**On Windows**:
1. Download PostgreSQL installer from postgresql.org
2. Run installer, note your password
3. Use pgAdmin or command line to create `practice_db`

### Step 6: Load Sample Data into PostgreSQL

```bash
# Run the database setup script
python scripts/setup_eu27_database_local.py

# Test connection
python scripts/test_postgres_connection.py
```

### Step 7: Run Streamlit App Locally

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

App will open at: `http://localhost:8501`

### Step 8: VS Code Workspace Layout

Recommended workspace layout:

```
┌─────────────────────────────────────────┐
│ File Explorer (Sidebar)                 │
│  ├── app.py                             │
│  ├── database_connection.py             │
│  ├── pages/                             │
│  │   ├── 09_🇪🇺_EU27_Database.py      │
│  │   ├── 12_🎯_Causal_Inference.py    │
│  │   └── ...                            │
│  ├── scripts/                           │
│  └── tests/                             │
└─────────────────────────────────────────┘
│ Editor (app.py open)                    │
└─────────────────────────────────────────┘
│ Terminal (running streamlit)            │
└─────────────────────────────────────────┘
```

**Keyboard shortcuts**:
- `Ctrl+`` - Toggle terminal
- `Ctrl+P` - Quick file open
- `Ctrl+Shift+P` - Command palette
- `F5` - Start debugging
- `Ctrl+/` - Toggle comment

## Working with Claude in VS Code

### Option 1: Continue Extension (Free, Supports Claude)

1. Install **Continue** extension
2. Press `Ctrl+Shift+P`, type "Continue: Add Model"
3. Select "Anthropic"
4. Enter your Claude API key
5. Use `Ctrl+L` to open Continue chat

### Option 2: GitHub Copilot (Paid)

1. Install **GitHub Copilot** extension
2. Sign in with GitHub account
3. Get inline suggestions as you type

### Option 3: Cody by Sourcegraph

1. Install **Cody** extension
2. Supports Claude models
3. Chat and autocomplete features

## Development Workflow

### Daily Workflow

```bash
# 1. Pull latest changes
git pull origin claude/streamlit-production-optimization-BvCxo

# 2. Create feature branch
git checkout -b feature/my-new-analysis

# 3. Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 4. Start PostgreSQL (if not running)
sudo systemctl start postgresql  # Linux
# or brew services start postgresql@16  # Mac

# 5. Run app in one terminal
streamlit run app.py

# 6. Code in VS Code editor

# 7. Test changes in browser (auto-refreshes)

# 8. Commit when done
git add .
git commit -m "Add new analysis feature"
git push -u origin feature/my-new-analysis
```

### Debugging Streamlit Apps

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Streamlit Debug",
      "type": "python",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "app.py", "--server.port", "8501"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

Now you can:
1. Set breakpoints in your code (click left of line number)
2. Press `F5` to start debugging
3. App pauses at breakpoints
4. Inspect variables in Debug sidebar

## Project Structure in VS Code

```
gender-wage-gap-analysis/
├── .vscode/                    # VS Code configuration
│   ├── settings.json           # Project settings
│   ├── launch.json             # Debug configurations
│   └── tasks.json              # Custom tasks
├── venv/                       # Virtual environment (git ignored)
├── app.py                      # 👈 Main entry point
├── database_connection.py      # 👈 Database utilities
├── sample_data.py              # Fallback data
├── requirements.txt            # Dependencies
├── .streamlit/
│   └── config.toml             # Streamlit config
├── pages/                      # 👈 Multi-page app pages
│   ├── 09_🇪🇺_EU27_Database.py
│   ├── 12_🎯_Causal_Inference.py
│   └── ...
├── scripts/                    # 👈 Analysis scripts
│   ├── time_series.py
│   ├── test_postgres_connection.py
│   └── ...
├── tests/                      # 👈 Unit tests
│   ├── test_app.py
│   └── ...
├── data/                       # Data files
│   └── processed/
├── DEPLOYMENT.md               # Deployment guide
├── INFRASTRUCTURE.md           # Infrastructure plan
└── VSCODE_SETUP.md            # This file
```

## Common Tasks

### Task 1: Add a New Page

```bash
# Create new page file
touch pages/18_📝_My_New_Page.py

# Use template:
# - Set page title with st.title()
# - Import required libraries
# - Add your analysis code
# - Test locally before committing
```

### Task 2: Update Database Schema

```python
# In scripts/migration_001_add_column.py
import psycopg2
from database_connection import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
    ALTER TABLE wage_gap_practice
    ADD COLUMN IF NOT EXISTS new_field NUMERIC;
""")

conn.commit()
cur.close()
conn.close()
```

### Task 3: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_app.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Task 4: Format Code

```bash
# Format all Python files
black .

# Check linting
flake8 .

# Type checking
mypy app.py
```

## Git Best Practices in VS Code

### Using GitLens

- **Line blame**: Hover over any line to see who wrote it and when
- **File history**: Click clock icon in editor title bar
- **Compare changes**: Right-click file → "Open Changes with Previous Revision"

### Commit Message Template

```
type(scope): short description

Detailed explanation of what changed and why

Fixes #123
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Example**:
```
feat(causal-inference): add propensity score matching

Implement PSM method for treatment effect estimation
- Add matching algorithm
- Add balance diagnostics
- Add visualization of matched pairs

Relates to PhD research Week 8
```

## Productivity Tips

### 1. Code Snippets

Create `.vscode/python.code-snippets`:

```json
{
  "Streamlit Page Template": {
    "prefix": "stpage",
    "body": [
      "import streamlit as st",
      "import pandas as pd",
      "import numpy as np",
      "import plotly.express as px",
      "",
      "st.set_page_config(page_title=\"${1:Page Title}\", page_icon=\"📊\", layout=\"wide\")",
      "",
      "st.title(\"${1:Page Title}\")",
      "st.markdown(\"---\")",
      "",
      "${2:# Your code here}",
      ""
    ],
    "description": "Streamlit page template"
  }
}
```

Type `stpage` + Tab to insert template.

### 2. Multi-Cursor Editing

- `Alt+Click` - Add cursor
- `Ctrl+Alt+↓` - Add cursor below
- `Ctrl+D` - Select next occurrence of current word
- `Ctrl+Shift+L` - Select all occurrences

### 3. Split Editor

- `Ctrl+\` - Split editor
- Work on `app.py` on left, `database_connection.py` on right

### 4. Integrated Terminal

- Open multiple terminals (PostgreSQL, Streamlit, testing)
- Name them by clicking dropdown → "Rename"

## Troubleshooting

### Issue: Python interpreter not found
**Fix**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `./venv/bin/python`

### Issue: Import errors in editor (red squiggles) but code runs
**Fix**: Reload window (`Ctrl+Shift+P` → "Reload Window")

### Issue: Streamlit not auto-reloading
**Fix**: Check "Always rerun" in top-right of Streamlit app

### Issue: PostgreSQL connection failed
**Fix**: Check service is running:
```bash
sudo systemctl status postgresql  # Linux
brew services list  # Mac
```

### Issue: Git authentication failed
**Fix**: Set up SSH keys or use Personal Access Token

## Next Steps After Setup

1. ✅ Clone repo
2. ✅ Install extensions
3. ✅ Set up Python environment
4. ✅ Configure PostgreSQL
5. ✅ Run app locally
6. 🔄 Start exploring the code
7. 🔄 Make small changes and test
8. 🔄 Read through the 17 pages to understand structure
9. 🔄 Review INFRASTRUCTURE.md for long-term plans

## Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
