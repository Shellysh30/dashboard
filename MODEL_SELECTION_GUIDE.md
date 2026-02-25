# Model Selection Feature Guide

## Overview

Your dashboard now supports filtering and comparing ROC curves by model! This is useful when you have multiple model versions or algorithms and want to compare their performance.

## Two Versions Available

### 1. `dashboard_new.py` (Standard)
- Single model selection dropdown
- Clean, simple interface
- Best for analyzing one model at a time
- Includes CSV download for ROC data

### 2. `dashboard_with_model_compare.py` (Advanced)
- Everything from standard version
- **PLUS:** Model comparison feature
- View all models on the same ROC curve
- Best for comparing multiple models side-by-side

## How It Works

### Automatic Detection

The dashboard automatically:
1. Checks if your BigQuery table has a model column
2. Loads unique model names
3. Creates a dropdown selector in the sidebar
4. Calculates separate ROC curves for each model

### If No Model Column Found

The dashboard will:
- Display a warning message
- Work normally without model filtering
- Show all predictions combined

## Configuration

### Step 1: Check Your Column Name

Your table might use different names for the model column. Common names:
- `model_name` (default)
- `model`
- `model_id`
- `model_version`
- `algorithm`
- `detector_name`
- `model_type`

### Step 2: Update the Configuration

Open `dashboard_new.py` and find line 16:

```python
MODEL_COLUMN_NAME = "model_name"  # Change this to match your column
```

Change it to your column name:

```python
MODEL_COLUMN_NAME = "model"  # Example
```

### Step 3: Restart the Dashboard

Stop (Ctrl+C) and restart:
```bash
streamlit run dashboard_new.py
```

## Using the Model Selector

### In the Sidebar

1. **🤖 Model Selection** section appears
2. Dropdown shows:
   - "All Models" (default - shows combined data)
   - Individual model names (sorted alphabetically)
3. **📊 Model Statistics** expander shows prediction counts

### What Changes When You Select a Model

When you select a specific model:
- ✅ Metrics update (Recall, FAR, TP, FP)
- ✅ ROC curve recalculates for that model only
- ✅ Chart title updates to show model name
- ✅ Dataset info shows model-specific counts
- ✅ Data preview filters to that model
- ✅ CSV download includes model name

### Example Workflow

1. Start with "All Models" to see overall performance
2. Select "Model_v1" to analyze first model
3. Note the Recall and FAR at threshold 0.5
4. Select "Model_v2" to compare
5. Compare metrics between models
6. Use advanced version for side-by-side comparison

## Model Comparison Feature

### Available in `dashboard_with_model_compare.py`

After selecting models individually, you can:

1. Check the **"Show model comparison"** checkbox
2. See all models overlaid on one ROC curve
3. Different colors for each model
4. Hover to see details for each model
5. Easily identify which model performs best

### What to Look For

**Better model has:**
- Higher recall at same FAR (curve is higher)
- Lower FAR at same recall (curve is more to the left)
- Curve closer to top-left corner

**Example:**
```
If Model A has 90% recall at FAR=0.05
And Model B has 85% recall at FAR=0.05
Then Model A is better (higher recall, same FAR)
```

## Troubleshooting

### "Column 'model_name' not found"

**Solution:** Update `MODEL_COLUMN_NAME` to match your table's column name

### "No data available for model: XYZ"

**Cause:** That model has no predictions or they were filtered out

**Solution:** 
- Check if the model name is spelled correctly
- Verify data exists for that model in BigQuery

### Model dropdown is empty

**Cause:** No model column found or all values are NULL

**Solution:**
- Verify the column exists: Run this in BigQuery console:
  ```sql
  SELECT DISTINCT model_name 
  FROM `mod-gcp-white-soi-dev-1.mantak_database.classified_predictions_third_eye`
  ```
- Update the column name if different

### Models not appearing in dropdown

**Cause:** Models might have NULL values

**Solution:** Add a WHERE clause in the query (line 50 of dashboard_new.py):
```python
WHERE confidence >= 0.0 AND {model_column} IS NOT NULL
```

## Performance Notes

### Caching

- Initial data load: Cached for 10 minutes (600 seconds)
- ROC calculations: Computed on-demand when you switch models
- Switching between models is fast (< 1 second)

### Large Datasets

If you have many models (> 10):
- Standard version works great
- Comparison version may be slow/cluttered
- Consider filtering in SQL query to specific models

### Optimization Tips

If the dashboard is slow:

1. **Reduce data limit** (line 42):
   ```python
   LIMIT 500000  # Reduce to 100000 for faster loading
   ```

2. **Pre-filter to specific models** (line 41):
   ```python
   WHERE confidence >= 0.0 AND model_name IN ('Model_v1', 'Model_v2')
   ```

3. **Reduce ROC points** (line 103):
   ```python
   thresholds = np.linspace(0, 1, 51)  # Reduce from 101 to 51
   ```

## Download ROC Data

For each model, you can:

1. Check "Show ROC curve data table"
2. Click "📥 Download ROC data as CSV"
3. File includes model name: `roc_data_Model_v1.csv`
4. Contains: threshold, recall, FAR, TP, FP for 101 points

Use this for:
- External analysis in Excel/Python
- Creating custom charts
- Reporting to stakeholders
- Archiving model performance

## Best Practices

### Analyzing Single Model
1. Select specific model
2. Adjust threshold slider to find optimal point
3. Note the recall/FAR trade-off
4. Download ROC data for documentation

### Comparing Multiple Models
1. Use `dashboard_with_model_compare.py`
2. Enable "Show model comparison"
3. Look for curves that dominate others (higher and to the left)
4. Select individual models to see exact numbers

### Finding Optimal Threshold
1. Start with "All Models" or specific model
2. Use slider to find acceptable FAR
3. Check if recall meets your requirements
4. Note the threshold value for deployment

### Reporting
1. Take screenshots of ROC curves
2. Download CSV data for tables
3. Note specific metrics at key thresholds
4. Compare models using overlay chart

## Example Use Cases

### Use Case 1: Version Comparison
```
Goal: Compare Model_v2 vs Model_v1
Steps:
1. Select Model_v1, note recall at FAR=0.01
2. Select Model_v2, note recall at FAR=0.01
3. Use comparison view to see both curves
4. Choose better performing model for deployment
```

### Use Case 2: Threshold Selection
```
Goal: Find threshold where FAR < 0.05 and Recall > 80%
Steps:
1. Select your model
2. Move slider until FAR shows < 0.05
3. Check if recall is > 80%
4. If not, relax FAR requirement or improve model
```

### Use Case 3: Algorithm Comparison
```
Goal: Compare YOLO vs Faster-RCNN vs SSD
Steps:
1. Ensure model_name contains algorithm names
2. Use comparison view
3. Identify which algorithm has best recall/FAR trade-off
4. Download data for detailed analysis
```

## Summary

✅ Model selection dropdown automatically appears if model column exists
✅ Configure column name with `MODEL_COLUMN_NAME` variable
✅ Compare models individually or side-by-side
✅ Download ROC data per model as CSV
✅ Works seamlessly even without model column

**Start analyzing your models now!** 🚀
