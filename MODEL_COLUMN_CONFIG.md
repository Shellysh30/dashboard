# Model Column Configuration

If your BigQuery table uses a different column name for the model, you can change it here:

## Option 1: Update the column name in dashboard_new.py

Open `dashboard_new.py` and find line 37:

```python
SELECT frame, confidence, eval_type, gt_id, model_name
```

Change `model_name` to whatever your column is called. Common alternatives:
- `model`
- `model_id`
- `model_version`
- `algorithm`
- `detector_name`

## Option 2: If your table doesn't have a model column

The dashboard will automatically detect this and work without model filtering.
It will show a warning: "No 'model_name' column found in data. Showing all predictions."

## Option 3: Add a model column using a SQL query

If you want to filter but don't have a model column, you can:

1. Add a static value in the query:
```sql
SELECT frame, confidence, eval_type, gt_id, 
       'YourModelName' as model_name
FROM `your_table`
```

2. Or derive it from another column:
```sql
SELECT frame, confidence, eval_type, gt_id,
       CONCAT('model_', version) as model_name
FROM `your_table`
```

## Testing

To see what columns are available in your table, you can run this query in BigQuery console:

```sql
SELECT * FROM `mod-gcp-white-soi-dev-1.mantak_database.classified_predictions_third_eye`
LIMIT 10
```

This will show you all available columns.
