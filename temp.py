import pandas as pd
from pandas_gbq import read_gbq

DATA_FILE = r"classified_predictions_third_eye.csv"

query = """
SELECT *
FROM    `mod-gcp-white-soi-dev-1.mantak_database.classified_predictions_third_eye`;
"""

df = pd.read_gbq(query, project_id='mod-gcp-white-soi-dev-1')
df.to_csv(DATA_FILE, index=False)
print("CSV saved successfully")
