# Sentiment_Analysis_VADER vs RoBERTa

### Project Overview
This project will perform sentiment analysis by comparing two models: VADER (a lexicon-based sentiment model) and RoBERTa (a pre-trained model for sentiment classification). The goal is to determine which model provides more accurate results and performs better.
### Data Source
The dataset used for this analysis was downloaded from Kaggle. It contains detailed information including but not limited to customer's age, price range, location and reviews. 
In total, it includes 15 columns and 7,394 rows.
<br/> 
Source: <a href="https://www.kaggle.com/datasets/adarsh0806/influencer-merchandise-sales">Kaggle, Merchandise Sales</a>

### Tools
* Python
    * matplotlib
    * numpy
    * seaborn 
    * tqdm
    * NLTK
       - VADER
            * `nltk.sentiment.SentimentIntensityAnalyzer`
       - RoBERTa   
            * transformers
            * scipy.special
    
### Running Model
 - VADER
   Is usually used for small-text 
 - RoBERTa 
   
### Accuracy
For measure model's accuracy I run regression metrics (MSE, MAE, Correlation), so I can evaluate how close the model's sentiment labels to human rating. 
      - from `sklearn.metrics` import `mean_squared_error`, `mean_absolute_error`
      - from `scipy.stats` import `pearsonr, spearmanr`
### Results

