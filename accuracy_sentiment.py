import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
plt.style.use("ggplot")
df=pd.read_excel("merch_1_with_sentiment.xlsx")
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

y_true = df['Rating'] 
y_pred_vader = df['vader_compound']  
y_pred_roberta = df['roberta_pos'] - df['roberta_neg'] 

# --- VADER metrics ---
mse_vader = mean_squared_error(y_true, y_pred_vader)
mae_vader = mean_absolute_error(y_true, y_pred_vader)
pearson_vader, _ = pearsonr(y_true, y_pred_vader)
spearman_vader, _ = spearmanr(y_true, y_pred_vader)

print("VADER Performance:")
print(f"MSE: {mse_vader:.4f}, MAE: {mae_vader:.4f}")
print(f"Pearson: {pearson_vader:.4f}, Spearman: {spearman_vader:.4f}\n")

# --- RoBERTa metrics ---
mse_roberta = mean_squared_error(y_true, y_pred_roberta)
mae_roberta = mean_absolute_error(y_true, y_pred_roberta)
pearson_roberta, _ = pearsonr(y_true, y_pred_roberta)
spearman_roberta, _ = spearmanr(y_true, y_pred_roberta)

print("RoBERTa Performance:")
print(f"MSE: {mse_roberta:.4f}, MAE: {mae_roberta:.4f}")
print(f"Pearson: {pearson_roberta:.4f}, Spearman: {spearman_roberta:.4f}")
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred_vader, alpha=0.5, label="VADER")
plt.scatter(y_true, y_pred_roberta, alpha=0.5, label="RoBERTa")
plt.xlabel("Human Ratings")
plt.ylabel("Predicted Sentiment Score")
plt.legend()
plt.show()