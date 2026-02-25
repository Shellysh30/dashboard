# 🚀 QUICK START GUIDE

## Start the Dashboard (Choose One Method)

### ⚡ Easiest Way
**Double-click:** `run_dashboard.bat`

### 💻 Command Line
```bash
streamlit run dashboard_new.py
```

### 🔧 PowerShell
```powershell
.\run_dashboard.ps1
```

---

## 📊 Using the Dashboard

### 1. Wait for Loading
The app will load data and pre-calculate 101 ROC points. This takes a moment but only happens once.

### 2. Adjust Confidence Threshold
Move the slider in the **left sidebar** from 0.0 to 1.0

### 3. Watch Metrics Update
- **Recall**: % of ground truth objects detected
- **FAR**: False alarms per frame  
- **TP**: True positive count
- **FP**: False positive count

### 4. Analyze ROC Curve
- **Hover** over points to see details
- **Zoom** in/out with mouse wheel
- **Pan** by clicking and dragging
- **Red dot** = your current threshold selection

### 5. Optional: View Data
Check the boxes at the bottom to see:
- Filtered predictions sample
- ROC curve data table

---

## 🛑 Stop the Dashboard
Press `Ctrl+C` in the terminal window

---

## ❓ Having Issues?

### "Could not connect to BigQuery"
→ Make sure `service-account-key.json` is in this folder

### "Module not found"
→ Run: `pip install -r requirements.txt`

### Dashboard won't open
→ Check terminal for error messages
→ Make sure no other app is using port 8501

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `dashboard_new.py` | Main dashboard code |
| `run_dashboard.bat` | Windows startup script |
| `requirements.txt` | Python dependencies |
| `service-account-key.json` | BigQuery credentials (you provide this) |
| `README.md` | Full documentation |
| `FIXES_APPLIED.md` | Technical details of fixes |

---

## 🎯 What You Can Do

✅ Find optimal confidence threshold for your model
✅ Balance recall vs false alarm rate
✅ Analyze trade-offs at different thresholds
✅ Export ROC curve data
✅ Share dashboard with team members

---

**Need Help?** Check `FIXES_APPLIED.md` for troubleshooting and technical details.
