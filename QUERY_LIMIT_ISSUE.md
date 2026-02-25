# 🚨 FOUND THE LIKELY ISSUE!

## The Problem

Your BigQuery shows for `third_eye_V0_11_epoch1.pt`:
```
WHERE confidence > 0.5:
- TP: 7,702
- D: 124
- FP: 118
- FP_IOU: 51
Total: 7,995 predictions
```

But dashboard shows:
```
- TP: 863
- FP: 0
Total: 863 predictions
```

## Root Cause: Query ORDER + LIMIT Issue

### Your Dashboard Query:
```sql
SELECT frame, confidence, eval_type, gt_id, model as model_name
FROM table
WHERE confidence >= 0.0
ORDER BY confidence DESC    ← Loads HIGHEST confidence first
LIMIT 500000               ← Stops after 500k rows
```

### What This Means:

1. **Loads highest confidence predictions first**
2. **Stops at 500,000 total rows across ALL models**
3. **If you have multiple models with high-confidence TPs, they fill the limit**
4. **Lower-confidence FPs get cut off!**

## Why You're Missing FPs

### Scenario:
```
All models combined, sorted by confidence DESC:

Row 1-400,000:   Model A, B, C - High confidence TPs (0.9-1.0)
Row 400,001-500,000: More high confidence TPs from other models
Row 500,001+:    Your FPs with confidence 0.5-0.8 ← NEVER LOADED!
```

The query hits the LIMIT before reaching your FPs!

### Your Specific Case:

For `third_eye_V0_11_epoch1.pt`:
- Dashboard loaded: 863 predictions (all TP, high confidence)
- BigQuery shows: 7,995 predictions total with confidence > 0.5
- **Missing: 7,132 predictions!** (including all FPs)

## Why BigQuery Query Works

Your test query:
```sql
WHERE model = 'third_eye_V0_11_epoch1.pt' 
AND confidence > 0.5
GROUP BY eval_type
```

This filters to ONE model first, THEN counts → sees all predictions for that model.

Dashboard query:
```sql
ORDER BY confidence DESC
LIMIT 500000  ← Cuts across ALL models
```

Then filters by model AFTER loading → misses low-confidence rows.

## The Fix

### Option 1: Increase LIMIT (Quick Fix)
Change line 41:
```python
LIMIT 500000  →  LIMIT 5000000  # or remove LIMIT entirely
```

⚠️ **Warning:** This loads more data, may be slower

### Option 2: Remove ORDER BY (Better)
```python
# Remove: ORDER BY confidence DESC
# Just:
LIMIT 500000
```

This loads random 500k rows, more likely to include FPs

### Option 3: Filter by Model in Query (Best)
```python
# Add model filter to query BEFORE ordering
WHERE confidence >= 0.0 
  AND model = 'your_model_name'  ← Filter first
ORDER BY confidence DESC
```

But this requires knowing the model ahead of time.

### Option 4: Sample Data (Fastest)
```python
# Load a sample across all confidence ranges
WHERE confidence >= 0.0
  AND MOD(ABS(FARM_FINGERPRINT(CAST(frame AS STRING))), 10) = 0
LIMIT 500000
```

This samples 10% of data randomly.

## Recommended Fix

**Change line 38-42 in dashboard_new.py:**

```python
# BEFORE:
query_with_model = f"""
    SELECT frame, confidence, eval_type, gt_id, model as model_name
    FROM `{FULL_TABLE_PATH}`
    WHERE confidence >= 0.0
    ORDER BY confidence DESC    ← REMOVE THIS
    LIMIT 500000
"""

# AFTER:
query_with_model = f"""
    SELECT frame, confidence, eval_type, gt_id, model as model_name
    FROM `{FULL_TABLE_PATH}`
    WHERE confidence >= 0.0
    LIMIT 2000000    ← Increase limit and remove ORDER BY
"""
```

Or even better, no limit at all if your dataset isn't huge:

```python
query_with_model = f"""
    SELECT frame, confidence, eval_type, gt_id, model as model_name
    FROM `{FULL_TABLE_PATH}`
    WHERE confidence >= 0.0
"""
```

## How to Verify

After the fix, run the dashboard and check:
1. **"Overall Data"** section should show 7,995+ predictions for your model
2. **Should see FP, FP_IOU, D** in eval_type breakdown
3. **Min confidence** should be closer to 0.0, not > 0.5

## Summary

✅ **Found**: ORDER BY + LIMIT cuts off low-confidence predictions
✅ **Cause**: High-confidence TPs from all models fill the 500k limit
✅ **Result**: Your FPs never get loaded
✅ **Fix**: Remove ORDER BY or increase LIMIT significantly

**This is why you see 863 instead of 7,995 predictions!**
