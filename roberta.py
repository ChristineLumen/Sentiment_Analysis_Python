import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
plt.style.use("ggplot")
df = pd.read_excel ("merch_1.xlsx", skiprows=1)
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

### BASIC NLTK
example = df["Review"][3]
tokens = nltk.word_tokenize(example)
tag=nltk.pos_tag(tokens)
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer =AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
score =sia.polarity_scores(example)
#Run for roberta model
roberta_neg =[]
roberta_neu =[]
roberta_pos =[]
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores =output.logits[0].detach().numpy()
    scores = softmax(scores)
    return {
    "roberta_neg":scores[0],
    "roberta_neu":scores[1],
    "roberta_pos":scores[2]
    }
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Review"]
    scores = polarity_scores_roberta(text)
    roberta_neg.append(scores["roberta_neg"])
    roberta_neu.append(scores["roberta_neu"])
    roberta_pos.append(scores["roberta_pos"])
df["roberta_neg"] = roberta_neg
df["roberta_neu"] = roberta_neu
df["roberta_pos"] = roberta_pos


df.to_excel("merch_1_with_sentiment_roberta.xlsx", index=False)
## Plot RoBERTa results
fig, axs = plt.subplots(1,3, figsize=(9,3))
ax= sns.barplot(data=df, x='Rating', y='roberta_pos', ax=axs[0])
axs[0].set_title("Positive Sentiment")
ax= sns.barplot(data=df, x='Rating', y='roberta_neu', ax=axs[2])
axs[2].set_title("Neutral Sentiment")
ax= sns.barplot(data=df, x='Rating', y='roberta_neg', ax=axs[1])
axs[1].set_title("Negative Sentiment")
plt.tight_layout()
plt.show()   
