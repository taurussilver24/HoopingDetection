import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import csv


# Function to generate the sample CSV


# Generate a new sample CSV file
# Load data from CSV into pandas DataFrame
df = pd.read_csv('test2.csv')

# Assuming your CSV has columns 'Actual Result' and 'Result'
# Here we're generating random scores for the sake of example
df['scores'] = [random.uniform(0, 1) for _ in range(len(df))]

# Define true labels based on 'Actual Result' column (e.g., Successful = 1, Failed = 0)
df['true_labels'] = df['Actual Result'].apply(lambda x: 1 if x == 'Successful' else 0)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(df['true_labels'], df['scores'])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
