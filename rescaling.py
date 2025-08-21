import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
plt.style.use("ggplot")
df = pd.read_excel ("merch_1.xlsx", skiprows=1)
###BASIC NLTK
example = df["Review"][3]
tokens = nltk.word_tokenize(example)
tag=nltk.pos_tag(tokens)

###VADER
from nltk.sentiment import SentimentIntensityAnalyzer 
from tqdm import tqdm
sia=SentimentIntensityAnalyzer()
df.to_excel("merch_1_with_sentiment.xlsx", index=False)
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer =AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
score =sia.polarity_scores(example)
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores =output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict ={
    "roberta_neg":scores[0],
    "roberta_neu":scores[1],
    "roberta_pos":scores[2]
    }
    return scores_dict
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Review"]
    
vader_result = sia.polarity_scores(text)
roberta_result = polarity_scores_roberta(text)
    
vader_neg_scores = []
vader_neu_scores = []
vader_pos_scores = []
compound_scores = []

roberta_neg_scores = []
roberta_neu_scores = []
roberta_pos_scores = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Review"]
    
    vader_result = sia.polarity_scores(text)
    vader_neg_scores.append(vader_result['neg'])
    vader_neu_scores.append(vader_result['neu'])
    vader_pos_scores.append(vader_result['pos'])
    compound_scores.append(vader_result['compound'])
    
    roberta_result = polarity_scores_roberta(text)
    roberta_neg_scores.append(roberta_result['roberta_neg'])
    roberta_neu_scores.append(roberta_result['roberta_neu'])
    roberta_pos_scores.append(roberta_result['roberta_pos'])

df['vader_neg'] = vader_neg_scores
df['vader_neu'] = vader_neu_scores
df['vader_pos'] = vader_pos_scores
df['vader_compound'] = compound_scores
df['roberta_neg'] = roberta_neg_scores
df['roberta_neu'] = roberta_neu_scores
df['roberta_pos'] = roberta_pos_scores

df.to_excel("merch_1_with_sentiment.xlsx", index=False)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

#VADER rescaling
y_pred_vader_scaled = 2 * (df['vader_compound'] + 1)  
y_pred_vader_scaled = y_pred_vader_scaled + 1         

#RoBERTa rescaling
y_pred_roberta_raw = df['roberta_pos'] - df['roberta_neg']
y_pred_roberta_scaled = 2 * (y_pred_roberta_raw + 1)  # 0..4
y_pred_roberta_scaled = y_pred_roberta_scaled + 1     # 1..5

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

y_true = df['Rating']

#VADER metrics
mse_vader = mean_squared_error(y_true, y_pred_vader_scaled)
mae_vader = mean_absolute_error(y_true, y_pred_vader_scaled)
pearson_vader, _ = pearsonr(y_true, y_pred_vader_scaled)
spearman_vader, _ = spearmanr(y_true, y_pred_vader_scaled)

print("VADER Performance (scaled):")
print(f"MSE: {mse_vader:.4f}, MAE: {mae_vader:.4f}")
print(f"Pearson: {pearson_vader:.4f}, Spearman: {spearman_vader:.4f}\n")

#RoBERTa metrics
mse_roberta = mean_squared_error(y_true, y_pred_roberta_scaled)
mae_roberta = mean_absolute_error(y_true, y_pred_roberta_scaled)
pearson_roberta, _ = pearsonr(y_true, y_pred_roberta_scaled)
spearman_roberta, _ = spearmanr(y_true, y_pred_roberta_scaled)

print("RoBERTa Performance (scaled):")
print(f"MSE: {mse_roberta:.4f}, MAE: {mae_roberta:.4f}")
print(f"Pearson: {pearson_roberta:.4f}, Spearman: {spearman_roberta:.4f}")
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred_vader_scaled, alpha=0.5, label="VADER")
plt.scatter(y_true, y_pred_roberta_scaled, alpha=0.5, label="RoBERTa")
plt.xlabel("Human Ratings")
plt.ylabel("Predicted Sentiment Score")
plt.legend()
plt.show()