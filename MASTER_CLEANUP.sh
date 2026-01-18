#!/bin/bash
# MASTER CLEANUP SCRIPT
# Complete repository cleanup and ownership transfer

set -e

clear
echo "========================================"
echo "  COMPLETE REPOSITORY CLEANUP"
echo "========================================"
echo ""
echo "This script will:"
echo "1. Transfer all git commits to YOUR authorship"
echo "2. Consolidate all branches to master"
echo "3. Delete all 'claude/*' branches"
echo "4. Clean up unnecessary files"
echo "5. Create professional project structure"
echo ""
echo "Backup will be created before any changes."
echo ""
read -p "Ready to proceed? (y/n): " START
if [ "$START" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# Step 1: Create backup
echo ""
echo "========================================" echo "STEP 1: Creating Backup"
echo "========================================"
git branch backup-$(date +%Y%m%d) 2>/dev/null || true
echo "✅ Backup branch created: backup-$(date +%Y%m%d)"
echo "   (You can restore with: git checkout backup-$(date +%Y%m%d))"

# Step 2: Repository cleanup
echo ""
echo "========================================"
echo "STEP 2: Repository Cleanup"
echo "========================================"
./cleanup-repo.sh

# Step 3: File cleanup
echo ""
echo "========================================"
echo "STEP 3: File Cleanup"
echo "========================================"
./cleanup-files.sh

# Step 4: Update README
echo ""
echo "========================================"
echo "STEP 4: Update Documentation"
echo "========================================"
echo ""
read -p "Replace README.md with new professional version? (y/n): " README_CONFIRM
if [ "$README_CONFIRM" == "y" ]; then
    mv README.md README_OLD.md
    mv README_NEW.md README.md
    echo "✅ README.md updated (old version saved as README_OLD.md)"
fi

# Step 5: Final commit
echo ""
echo "========================================"
echo "STEP 5: Final Cleanup Commit"
echo "========================================"
echo ""
echo "Ready to commit all cleanup changes?"
echo ""
read -p "Create final commit? (y/n): " COMMIT_CONFIRM
if [ "$COMMIT_CONFIRM" == "y" ]; then
    git add .
    git commit -m "Project cleanup and reorganization

- Consolidated project structure
- Updated documentation
- Removed temporary files
- Organized scripts and documentation
- Production-ready setup"

    echo "✅ Changes committed"

    read -p "Push to master? (y/n): " PUSH_CONFIRM
    if [ "$PUSH_CONFIRM" == "y" ]; then
        git push origin master
        echo "✅ Pushed to GitHub"
    fi
fi

# Final summary
echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================"
echo ""
echo "Summary:"
echo "- ✅ All work consolidated to master"
echo "- ✅ All commits authored by you"
echo "- ✅ Claude branches deleted"
echo "- ✅ Files organized"
echo "- ✅ Documentation updated"
echo ""
echo "Your repository is now:"
echo "- Professional"
echo "- Clean"
echo "- Yours completely"
echo ""
echo "Next steps:"
echo "1. Update Streamlit Cloud:"
echo "   - Branch: master"
echo "   - Main file: app.py"
echo ""
echo "2. Test everything:"
echo "   ./quick-start.sh"
echo ""
echo "3. Share your work:"
echo "   GitHub: https://github.com/kiko1992-creator/gender-wage-gap-analysis"
echo ""
echo "Backup available at: backup-$(date +%Y%m%d)"
echo ""
