#!/bin/bash
# File Cleanup Script
# Identifies and removes unnecessary files

echo "========================================"
echo "File Cleanup Wizard"
echo "========================================"
echo ""

# Create archive directory
mkdir -p archive/old-docs
mkdir -p archive/old-scripts
mkdir -p docs

echo "File Analysis:"
echo ""

# Documentation files
echo "📚 Documentation files found:"
find . -maxdepth 1 -name "*.md" -type f | grep -v README.md
echo ""

echo "Recommended actions:"
echo "1. Keep: README.md (main docs)"
echo "2. Move to docs/: DOCKER_GUIDE.md, DEPLOYMENT.md, INFRASTRUCTURE.md, VSCODE_SETUP.md"
echo "3. Delete: NEXT_STEPS.md, CLEANUP_PLAN.md (temporary)"
echo ""

read -p "Consolidate documentation? (y/n): " DOCS_CONFIRM
if [ "$DOCS_CONFIRM" == "y" ]; then
    # Move useful docs to docs/ folder
    mv DOCKER_GUIDE.md docs/ 2>/dev/null || true
    mv DEPLOYMENT.md docs/ 2>/dev/null || true
    mv INFRASTRUCTURE.md docs/ 2>/dev/null || true
    mv VSCODE_SETUP.md docs/ 2>/dev/null || true

    # Delete temporary docs
    rm -f NEXT_STEPS.md
    rm -f CLEANUP_PLAN.md

    echo "✅ Documentation consolidated"
fi

echo ""
echo "📁 Scripts directory:"
ls -lh scripts/ | head -20
echo ""

echo "Scripts to keep:"
echo "  - time_series.py (used by app)"
echo "  - test_postgres_connection.py (testing)"
echo "  - setup_eu27_database_local.py (database)"
echo ""

echo "Scripts to archive (not actively used):"
ls scripts/*.py | while read script; do
    basename="$(basename $script)"
    if [[ "$basename" != "time_series.py" ]] && \
       [[ "$basename" != "test_postgres_connection.py" ]] && \
       [[ "$basename" != "setup_eu27_database_local.py" ]] && \
       [[ "$basename" != "__init__.py" ]]; then
        echo "  - $basename"
    fi
done

echo ""
read -p "Archive unused scripts? (y/n): " SCRIPTS_CONFIRM
if [ "$SCRIPTS_CONFIRM" == "y" ]; then
    ls scripts/*.py | while read script; do
        basename="$(basename $script)"
        if [[ "$basename" != "time_series.py" ]] && \
           [[ "$basename" != "test_postgres_connection.py" ]] && \
           [[ "$basename" != "setup_eu27_database_local.py" ]] && \
           [[ "$basename" != "__init__.py" ]]; then
            mv "$script" archive/old-scripts/ 2>/dev/null || true
        fi
    done

    # Archive markdown files in scripts
    mv scripts/*.md archive/old-docs/ 2>/dev/null || true

    echo "✅ Scripts archived"
fi

echo ""
echo "🗑️  Removing .private directory (sensitive/temp files)..."
read -p "Delete .private/ directory? (y/n): " PRIVATE_CONFIRM
if [ "$PRIVATE_CONFIRM" == "y" ]; then
    rm -rf .private/
    echo "✅ .private/ deleted"
fi

echo ""
echo "📦 Test files:"
ls -lh tests/ 2>/dev/null | head -10 || echo "No tests directory"
echo ""

read -p "Keep tests/ directory? (y/n): " TESTS_CONFIRM
if [ "$TESTS_CONFIRM" != "y" ]; then
    mv tests/ archive/ 2>/dev/null || true
    echo "✅ Tests archived"
fi

echo ""
echo "🗂️  Output directory:"
du -sh output/ 2>/dev/null || echo "No output directory"
echo ""

read -p "Archive output/ directory? (y/n): " OUTPUT_CONFIRM
if [ "$OUTPUT_CONFIRM" == "y" ]; then
    mv output/ archive/ 2>/dev/null || true
    echo "✅ Output archived"
fi

# Update .gitignore
echo ""
echo "Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Archived files
archive/

# Environment
.env

# IDE
.vscode/
.idea/

# Python
__pycache__/
*.pyc
.pytest_cache/

# Data
data/raw/
*.csv
*.xlsx

# Temporary
*.log
.DS_Store
EOF

echo "✅ .gitignore updated"

echo ""
echo "========================================"
echo "Final project structure:"
echo "========================================"
echo ""

tree -L 2 -I '__pycache__|.git|archive|.pytest_cache' || ls -la

echo ""
echo "========================================"
echo "✅ FILE CLEANUP COMPLETE!"
echo "========================================"
echo ""
echo "Summary:"
echo "- Documentation moved to docs/"
echo "- Unused scripts archived"
echo "- Temporary files removed"
echo "- .gitignore updated"
echo ""
echo "Next steps:"
echo "1. Review the changes"
echo "2. Test app still works: ./quick-start.sh"
echo "3. Commit changes:"
echo "   git add ."
echo "   git commit -m 'Clean up project structure'"
echo "   git push origin master"
echo ""
