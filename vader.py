import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
plt.style.use("ggplot")
df = pd.read_excel ("merch_1.xlsx", skiprows=1)
example = df["Review"][3]
tokens = nltk.word_tokenize(example)
tag=nltk.pos_tag(tokens)

### VADER
from nltk.sentiment import SentimentIntensityAnalyzer 
from tqdm import tqdm
sia=SentimentIntensityAnalyzer()
score =sia.polarity_scores(example)
print(score)
neg_scores = []
neu_scores = []
pos_scores = []
compound_scores = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Review']
    sentiment = sia.polarity_scores(text)
    neg_scores.append(sentiment['neg'])
    neu_scores.append(sentiment['neu'])
    pos_scores.append(sentiment['pos'])
    compound_scores.append(sentiment['compound'])
df['neg'] = neg_scores
df['neu'] = neu_scores
df['pos'] = pos_scores
df['compound'] = compound_scores


df.to_excel("merch_1_with_sentiment.xlsx", index=False)
## Plot VADER results
fig, axs = plt.subplots(1,3, figsize=(9,3))
ax= sns.barplot(data=df, x='Rating', y='pos', ax=axs[0])
axs[0].set_title("Positive Sentiment")
ax= sns.barplot(data=df, x='Rating', y='neu', ax=axs[2])
axs[2].set_title("Neutral Sentiment")
ax= sns.barplot(data=df, x='Rating', y='neg', ax=axs[1])
axs[1].set_title("Negative Sentiment")
plt.tight_layout()
plt.show()
