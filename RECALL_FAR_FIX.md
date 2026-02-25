# ✅ Fixed: Recall and FAR Calculations

## What Was Wrong

### Before (Incorrect):
```python
# Wrong recall calculation
detected_gt = df_filtered.loc[temp_filtered_tp, 'gt_id'].nunique()
recall = detected_gt / total_gt
```

**Problem:** This counted UNIQUE ground truth IDs, not the actual TP/FN relationship.

### After (Correct - Matches Your Code):
```python
# Correct recall calculation
# 1. Get all ground truth annotations that should be detected
gt_annotation_set = set of (scenario, frame, gt_id) tuples

# 2. For each threshold, find which GTs were detected
detected_gt_set = set of (scenario, frame, gt_id) for TP predictions

# 3. Calculate FN = ground truth NOT detected
fn_set = gt_annotation_set - detected_gt_set
fn_count = len(fn_set)

# 4. Recall = TP / (TP + FN)
recall = tp_count / (tp_count + fn_count)
```

## Key Differences

### Recall Calculation:

| Aspect | Wrong Method | Correct Method |
|--------|-------------|----------------|
| Formula | `unique_detected_gts / total_unique_gts` | `TP / (TP + FN)` |
| Counts | Unique GT IDs only | Actual predictions |
| FN Source | Implicit (missing GTs) | Explicit calculation |
| Matches Your Code | ❌ No | ✅ Yes |

### FAR Calculation:

Both methods use the same calculation (this was correct):
```python
FAR = FP / total_frames
```

## Example Showing the Difference

### Scenario:
```
Frame 1, GT Object #1:
  - Detection A: confidence=0.9, TP
  - Detection B: confidence=0.8, TP (duplicate)
  
Frame 2, GT Object #2:
  - No detections (FN)

Total GT annotations: 2 (Frame 1 GT#1, Frame 2 GT#2)
```

### Wrong Method (Before):
```python
Threshold 0.5:
  detected_unique_gts = 1 (only GT#1)
  total_unique_gts = 2 (GT#1, GT#2)
  recall = 1/2 = 50%
```

### Correct Method (After):
```python
Threshold 0.5:
  TP predictions = 2 (Detection A, B)
  FN = 1 (Frame 2 GT#2 not detected)
  recall = 2 / (2 + 1) = 66.7%
```

**The correct method gives higher recall** because it counts actual TP predictions, not just unique objects!

## What Changed in Code

### 1. Added `scenario` field to query:
```python
SELECT frame, confidence, eval_type, gt_id, model as model_name, scenario
```
Needed for (scenario, frame, gt_id) tuple identification.

### 2. Changed recall calculation logic:
```python
# Build set of all GT annotations
gt_annotations = df_filtered[['scenario', 'frame', 'gt_id']].drop_duplicates()
gt_annotation_set = set(gt_annotations.itertuples(index=False, name=None))

# For each threshold:
tp_df = df_filtered[tp_mask][['scenario', 'frame', 'gt_id']].drop_duplicates()
detected_gt_set = set(tp_df.itertuples(index=False, name=None))

# Calculate FN
fn_set = gt_annotation_set - detected_gt_set
fn_count = len(fn_set)

# Recall = TP / (TP + FN)
recall = tp_count / (tp_count + fn_count)
```

### 3. Added FN metric to display:
Now shows 5 metrics instead of 4:
- Recall
- FAR  
- TP
- FP
- **FN** (new!)

## Formula Summary

### Your Correct Code:
```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
FAR = FP / total_frames
```

### Dashboard Now Matches:
```python
recall = tp_count / (tp_count + fn_count)  ✅
far = fp_count / num_frames  ✅
```

## Benefits

✅ **Accurate recall** - Matches your reference implementation
✅ **Shows FN** - See how many GTs were missed
✅ **Proper TP/FN balance** - Counts predictions, not just unique IDs
✅ **Standard metric** - Follows ML best practices

## Performance Note

The calculation is still fast because:
- Uses set operations (O(n) complexity)
- Drop_duplicates() is optimized in pandas
- Only 51 threshold points to calculate

## Test It

```bash
streamlit run dashboard_new.py
```

You should now see:
- Correct recall values (likely higher than before)
- FN count displayed
- Recall = TP / (TP + FN) ✅

**Recall and FAR now calculated correctly!** 🎯
