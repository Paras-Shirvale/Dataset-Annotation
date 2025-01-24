import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the CSV file with two annotators
df_nlp = pd.read_csv('nlp_dataset.csv')

# Assuming the CSV has columns 'annotator_1' and 'annotator_2'
annotator_1 = df_nlp['annotator_1'].values
annotator_2 = df_nlp['annotator_2'].values

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's Kappa: {kappa}")
