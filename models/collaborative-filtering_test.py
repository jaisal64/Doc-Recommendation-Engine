import pandas as pd

patients_df = pd.read_csv('/data/patients_data.csv')
doctors_df = pd.read_csv('/data/diverse_doctors_data.csv')

doctors_df['cost_min'] = doctors_df['cost_min'].replace('[\$,]', '', regex=True).astype(float)
doctors_df['cost_max'] = doctors_df['cost_max'].replace('[\$,]', '', regex=True).astype(float)
patients_df['budget_max'] = patients_df['budget_max'].replace('[\$,]', '', regex=True).astype(float)


