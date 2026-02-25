# Troubleshooting: "0 FP" Despite Having FP in Data

## Your Current Issue

You see:
- TP: 863
- FP: 0
- Warning: "No False Positives found!"
- Vertical ROC curve

But you KNOW there are FPs in your BigQuery data for this model.

## Possible Causes

### 1. **Model Name Mismatch** (Most Likely)
The model name in the dropdown might not exactly match what's in BigQuery.

**Check for:**
- Extra spaces: `"model_v1 "` vs `"model_v1"`
- Case differences: `"Model_V1"` vs `"model_v1"`
- Underscores vs dots: `"third_eye_V0.11"` vs `"third_eye_V0_11"`
- File extensions: `"model.pt"` vs `"model"`

### 2. **All FPs Filtered Out by Confidence**
If all FP predictions have confidence < 0.0, they won't be loaded.

The query has: `WHERE confidence >= 0.0`

### 3. **FPs Belong to Different Model**
The FPs you see in BigQuery might be for a different model name.

### 4. **Data Issue in BigQuery**
For this specific model, maybe all predictions really ARE marked as TP.

## New Diagnostics Added

### 1. **Overall Data Stats**
Expandable section showing:
- All models and their prediction counts
- Eval types in entire dataset BEFORE filtering

### 2. **Model Filtering Check**
Shows:
- Selected model name (exact string)
- Actual model values in filtered data
- Similar model names that might be confused

### 3. **Eval Type Check**
Shows:
- All unique eval_type values found
- Exact strings for each eval type

### 4. **Sample Data**
First 50 rows with eval_type visible

## What to Do Now

### Step 1: Run the dashboard
```bash
streamlit run dashboard_new.py
```

### Step 2: Check "Overall Data (Before Model Filter)"
Expand this section and answer:
- ✅ Is `third_eye_V0_11_epoch1.pt` listed?
- ✅ What's the exact spelling/capitalization?
- ✅ Does it have FP/FP_IOU/D in overall dataset?

### Step 3: Check "Model Filtering Check"
When you select the model, this shows:
- What you selected
- What's actually in the filtered data
- Similar model names

### Step 4: Check "All Eval Types in This Selection"
This shows EXACTLY what eval_types exist after filtering

### Step 5: Look at Sample Data
First 50 rows - do you see any non-TP eval_types?

## Expected Scenarios

### Scenario A: Model Name Mismatch
```
Overall Data shows: third_eye_V0_11_epoch1.pt (1,500 rows, has FP)
You selected: third_eye_V0_11_epoch1.pt
Filtered data: 0 rows or different model name

→ Problem: Exact name doesn't match
→ Solution: Copy exact name from "Overall Data" section
```

### Scenario B: Wrong Model
```
Overall Data shows:
  - third_eye_V0_11_epoch1.pt: 863 rows (all TP)
  - third_eye_V0_11_epoch2.pt: 1,200 rows (has FP)

→ Problem: You selected epoch1, FPs are in epoch2
→ Solution: Select the correct model
```

### Scenario C: All FPs Filtered Out
```
Overall Data shows: Model has FP_IOU with confidence < 0
Query filters: WHERE confidence >= 0.0

→ Problem: FPs have negative confidence
→ Solution: Change query to LIMIT confidence >= -1.0 or no limit
```

### Scenario D: Data Really Has No FPs
```
Overall Data shows: Model has 863 TP, 0 FP

→ This model truly has no false positives
→ Vertical ROC is correct behavior
```

## Quick Fix: Check BigQuery Directly

Run this in BigQuery console:
```sql
SELECT 
    model,
    eval_type,
    COUNT(*) as count
FROM `mod-gcp-white-soi-dev-1.mantak_database.classified_predictions_third_eye`
WHERE model LIKE '%third_eye_V0_11%'
GROUP BY model, eval_type
ORDER BY model, eval_type
```

This shows EXACTLY what's in BigQuery for this model.

## If Problem Persists

Share the output of:
1. **Overall Data (Before Model Filter)** section
2. **Model Filtering Check** section
3. **All Eval Types in This Selection** section

This will help identify the exact issue.

## Most Common Issue

**95% of the time it's:** Model name has extra characters or different spelling than what you think.

**Solution:** Use the exact name from "Overall Data" dropdown, copy-paste if needed.
