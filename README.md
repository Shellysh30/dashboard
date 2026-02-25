# Third Eye Model Evaluation Dashboard

Interactive Streamlit dashboard for analyzing model performance with ROC curves, recall, and FAR metrics.

## Features

- 📊 **Interactive ROC Curve** - Beautiful Plotly visualization with hover details
- 🤖 **Model Selection** - Filter ROC curves by specific models
- 🎚️ **Confidence Threshold Slider** - Adjust threshold from 0.0 to 1.0 in real-time
- 📈 **Recall & FAR Calculations** - Automatic calculation at each threshold
- ⚡ **Pre-calculated Metrics** - Fast performance with 101 pre-computed ROC points
- 📋 **Data Preview** - View filtered predictions and ROC data table
- 🎯 **Real-time Updates** - Metrics update instantly as you move the slider
- 🔄 **Model Comparison** - Compare ROC curves across multiple models (optional)

## Installation

### Quick Setup

1. **Install dependencies** (if not already installed):
```bash
pip install -r requirements.txt
```

2. **Setup BigQuery credentials**:
   - Option A: Place `service-account-key.json` in the DASHBOARD folder
   - Option B: Use default application credentials (if configured)

## Usage

### Windows Users

**Easy Start** - Double-click one of these files:
- `run_dashboard.bat` - For Command Prompt
- `run_dashboard.ps1` - For PowerShell

**Manual Start:**
```bash
streamlit run dashboard_new.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### To Stop the Server
Press `Ctrl+C` in the terminal

## Model Selection Feature

The dashboard now supports filtering by model:

### If your data has a model column

The dashboard will automatically detect it and show a **Model Selection** dropdown in the sidebar. You can:
- Select "All Models" to see combined data
- Select a specific model to see its individual performance
- View model statistics (prediction counts per model)

### If your column name is different

Edit line 16 in `dashboard_new.py`:
```python
MODEL_COLUMN_NAME = "model_name"  # Change to your column name
```

Common alternatives: `model`, `model_id`, `model_version`, `algorithm`, `detector_name`

### Advanced: Model Comparison

Use `dashboard_with_model_compare.py` for additional features:
- Compare multiple models on the same ROC curve
- See all models overlaid for easy comparison
- Better for analyzing multiple model versions

Run it with:
```bash
streamlit run dashboard_with_model_compare.py
```

## Features Explained

### Metrics
- **Recall**: Percentage of ground truth objects detected at the selected threshold
- **FAR (False Alarm Rate)**: Number of false positives per frame
- **True Positives (TP)**: Correct detections
- **False Positives (FP)**: Incorrect detections (false alarms)

### ROC Curve
- X-axis: FAR (False Alarm Rate)
- Y-axis: Recall
- Color scale: Confidence threshold
- Red dot: Currently selected threshold

### Confidence Threshold Slider
Adjust the slider in the sidebar to see how metrics change at different confidence thresholds.
