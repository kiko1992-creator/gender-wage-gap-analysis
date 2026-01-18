#!/bin/bash
# Repository Cleanup Script
# Consolidates all work to master branch with YOUR authorship

set -e

echo "========================================"
echo "Repository Cleanup & Ownership Transfer"
echo "========================================"
echo ""

# Get user information
read -p "Enter your full name (e.g., Kiril Mickovski): " USER_NAME
read -p "Enter your email (e.g., your@email.com): " USER_EMAIL

echo ""
echo "Setting git configuration..."
git config user.name "$USER_NAME"
git config user.email "$USER_EMAIL"

echo "✅ Git configured:"
echo "   Name: $USER_NAME"
echo "   Email: $USER_EMAIL"
echo ""

# Confirm
read -p "This will consolidate all branches to master. Continue? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "📋 Current branches:"
git branch -a | grep -v "remotes"
echo ""

# Create backup branch
echo "Creating backup branch..."
git branch backup-before-cleanup || true

# Checkout master and update
echo ""
echo "Switching to master..."
git checkout master
git pull origin master || true

# Get all changes from feature branch
echo ""
echo "Merging all work from claude/streamlit-production-optimization-BvCxo..."
git merge --squash claude/streamlit-production-optimization-BvCxo

# Create clean commit with your authorship
echo ""
echo "Creating clean commit..."
git commit -m "Complete PhD research platform with Docker automation

Comprehensive econometrics learning platform with:
- 17 interactive pages (Causal Inference, Panel Econometrics, ML, Bayesian, Time Series)
- Docker infrastructure (PostgreSQL + Streamlit)
- Production-ready deployment
- Database automation
- Interactive tutorials

Built for PhD preparation in econometrics research on gender wage gap analysis."

echo "✅ Clean commit created by $USER_NAME"
echo ""

# Show commit
git log -1 --format="%h %an <%ae> - %s"

echo ""
read -p "Push to master on GitHub? (y/n): " PUSH_CONFIRM

if [ "$PUSH_CONFIRM" == "y" ]; then
    echo "Pushing to master..."
    git push origin master
    echo "✅ Pushed to master"
else
    echo "Skipped push. Run 'git push origin master' when ready."
fi

echo ""
echo "========================================" echo "Local branch cleanup..."
echo "========================================"
echo ""
echo "Deleting local claude branches..."

git branch -D claude/find-fix-bug-mjst54m7xyf7wwf3-a3xmE || true
git branch -D claude/fix-timeseries-BvCxo || true
git branch -D claude/merge-master-fix || true
git branch -D claude/postgres-tutorial-setup-BvCxo || true
git branch -D claude/streamlit-production-optimization-BvCxo || true

echo "✅ Local branches cleaned"
echo ""

read -p "Delete remote claude branches on GitHub? (y/n): " DELETE_REMOTE
if [ "$DELETE_REMOTE" == "y" ]; then
    echo "Deleting remote branches..."

    git push origin --delete claude/find-fix-bug-mjst54m7xyf7wwf3-a3xmE || true
    git push origin --delete claude/fix-timeseries-BvCxo || true
    git push origin --delete claude/postgres-tutorial-setup-BvCxo || true
    git push origin --delete claude/streamlit-production-optimization-BvCxo || true
    git push origin --delete streamlit-production-optimization || true
    git push origin --delete timeseries-BvCxo || true

    echo "✅ Remote branches deleted"
else
    echo "Skipped remote branch deletion."
    echo "Delete manually on GitHub if needed."
fi

echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================"
echo ""
echo "Repository state:"
git branch -a
echo ""
echo "Latest commit:"
git log -1 --oneline
echo ""
echo "Next steps:"
echo "1. Review CLEANUP_PLAN.md for file cleanup"
echo "2. Delete unnecessary files"
echo "3. Update Streamlit Cloud to use 'master' branch"
echo "4. Test everything still works"
echo ""
