# How to Rebuild/Reopen DevContainer in VS Code

## ✅ What Was Added

Your `.devcontainer/devcontainer.json` now includes:

1. **Claude Code VS Code Extension** (`anthropics.claude-code`)
   - Automatically installs when container builds

2. **Claude CLI**
   - Installs via official installation script
   - Runs during `postCreateCommand`
   - Available after container creation

## 🔄 How to Rebuild Container

### **Method 1: Command Palette (Recommended)**

1. Open VS Code in your project
2. Press **`F1`** or **`Ctrl+Shift+P`** (Windows/Linux) / **`Cmd+Shift+P`** (Mac)
3. Type: **`Dev Containers: Rebuild Container`**
4. Press Enter
5. Wait 2-5 minutes for rebuild

### **Method 2: Rebuild Without Cache (Clean Rebuild)**

1. Press **`F1`** or **`Ctrl+Shift+P`**
2. Type: **`Dev Containers: Rebuild Container Without Cache`**
3. Press Enter
4. Wait 3-7 minutes (downloads everything fresh)

### **Method 3: Reopen in Container (If Already Built)**

1. Press **`F1`** or **`Ctrl+Shift+P`**
2. Type: **`Dev Containers: Reopen in Container`**
3. Press Enter
4. Wait 30-60 seconds

## ✓ Verify Claude CLI Installation

After container rebuilds, open terminal in VS Code:

```bash
# Check Claude CLI version
claude --version

# Should output something like:
# claude-cli 1.x.x
```

If command not found, reload terminal:

```bash
# Reload shell
source ~/.bashrc
# or
source ~/.zshrc

# Try again
claude --version
```

## 🔍 What Happens During Rebuild

1. **Base Image**: Downloads Python 3.11 container
2. **Extensions Install**:
   - Python extension
   - Pylance (Python language server)
   - **Claude Code extension** ⭐ (new)
3. **updateContentCommand**:
   - Updates packages
   - Installs Python requirements
   - Installs Streamlit
4. **postCreateCommand**:
   - **Installs Claude CLI** ⭐ (new)
   - Verifies installation
5. **postAttachCommand**:
   - Starts Streamlit app on port 8501

## 🎯 Quick Test After Rebuild

```bash
# Test Claude CLI
claude --version

# Test Python
python --version

# Test project runs
streamlit run app.py
```

## 🐛 Troubleshooting

### Issue: Claude CLI not found

**Solution 1 - Reload Terminal:**
```bash
source ~/.bashrc
claude --version
```

**Solution 2 - Check Installation:**
```bash
ls -la ~/bin/claude
# or
which claude
```

**Solution 3 - Manual Install:**
```bash
curl -fsSL https://raw.githubusercontent.com/anthropics/claude-code/main/install.sh | sh
source ~/.bashrc
```

### Issue: Extensions not installing

**Solution:**
1. Rebuild without cache (Method 2 above)
2. Check VS Code Extensions panel (Ctrl+Shift+X)
3. Manually search for "Claude Code" and install

### Issue: Container won't rebuild

**Solution:**
1. Close VS Code
2. Delete container: `docker rm -f <container-name>`
3. Reopen VS Code in project
4. Rebuild container

## 📋 Rebuild Checklist

After rebuilding, verify:

- [ ] VS Code reopens in container
- [ ] Terminal shows container shell
- [ ] `python --version` shows 3.11.x
- [ ] `claude --version` works
- [ ] Claude Code extension visible in Extensions panel
- [ ] `streamlit run app.py` starts app
- [ ] App accessible at http://localhost:8501

## ⚡ Quick Command Reference

```bash
# Inside VS Code Command Palette (F1):

Dev Containers: Rebuild Container          # Normal rebuild
Dev Containers: Rebuild Without Cache      # Clean rebuild
Dev Containers: Reopen in Container        # Quick reopen
Dev Containers: Reopen Folder Locally      # Exit container

# Inside container terminal:

claude --version                           # Check Claude CLI
python --version                           # Check Python
streamlit run app.py                       # Run app
docker ps                                  # See running containers
```

## 🎓 Understanding DevContainer Lifecycle

```
VS Code Opens Project
       ↓
Detects .devcontainer/devcontainer.json
       ↓
Asks: "Reopen in Container?"
       ↓
[Yes] → Builds/Starts Container
       ↓
Runs: updateContentCommand (packages)
       ↓
Runs: postCreateCommand (Claude CLI install)
       ↓
Runs: postAttachCommand (Streamlit start)
       ↓
Container Ready! 🎉
```

## 💡 Pro Tips

1. **First time build is slow** - Container downloads ~500MB base image
2. **Subsequent rebuilds are faster** - Uses cached layers
3. **Without Cache rebuild** - Use only if things are broken
4. **Container persists data** - Your files are safe during rebuilds
5. **Extension installs automatically** - No manual installation needed

---

**Next Step:** Rebuild your container using Method 1 above and verify `claude --version` works!
