import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
plt.style.use("ggplot")
df = pd.read_excel ("merch_1.xlsx", skiprows=1)
# print(df.head())
# print(df["Review"].values[0])
#print(df.shape)
# ax= df["Rating"].value_counts().sort_index().plot(kind='bar', 
#                                                     title = "Count of reviews by Rating"
#                                                     )
# ax.set_xlabel("Review Stars")
# plt.xticks(rotation =0)
# plt.show()

### BASIC NLTK
#example = df["Review"][50]
# tokens = nltk.word_tokenize(example)
# tag=nltk.pos_tag(tokens)

### VADER
from nltk.sentiment import SentimentIntensityAnalyzer 
from tqdm import tqdm
sia=SentimentIntensityAnalyzer()
# score =sia.polarity_scores(example)
# print(score)
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

# Add new columns to your DataFrame
df['neg'] = neg_scores
df['neu'] = neu_scores
df['pos'] = pos_scores
df['compound'] = compound_scores

# Save the updated DataFrame to a new Excel file (or overwrite existing)
df.to_excel("merch_1_with_sentiment.xlsx", index=False)
# Plot VADER results
ax= sns.barplot(data=df, x='Rating', y='compound')
ax.set_title("Compound Score")
plt.show()
# avg_scores =df[['neg','pos', 'compound']].mean().reset_index()
# avg_scores.columns=['Sentiment', 'Average Score']
# plt.figure(figsize= (8,5))
# sns.barplot(x='Sentiment', y='Average Score', data=avg_scores)
# plt.title("Average VADER Sentiment Scores")
# plt.ylim(0, 1)
# plt.show()



# nltk.download("vader_lexicon")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download('words')
# from nltk.tree import Tree
# from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
# from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# tqdm.pandas()
# sia_analyzer=sia()
# df = pd.read_excel("merch_1.xlsx", skiprows=1)

# df["Review_Tokens"] = df["Review"].apply(lambda x: word_tokenize(str(x)))
# df["POS_Tags"] = df["Review_Tokens"].apply(lambda tokens: nltk.pos_tag(tokens))
# def remove_in_tags(tagged_tokens):
#     return[(word,tag)for word, tag in tagged_tokens if tag!= "IN"]
# df["POS_Tags_No_IN"] =df["POS_Tags"].apply(remove_in_tags)
# df["Polarity_Score"]=df["Review"].progress_apply(lambda text:sia_analyzer.polarity_scores(str(text)))
# print(df["Polarity_Score"].head())

