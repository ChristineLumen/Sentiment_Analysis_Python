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
For measuring model accuracy, I used regression metrics (MSE, MAE, Pearson and Spearman correlations) to evaluate how close the model’s sentiment predictions are to human ratings:
* <b>Size of errors</b> – how far off the predictions are from actual ratings
      * `sklearn.metrics.mean_squared_error`
      * `sklearn.metrics.mean_absolute_error`
* <b>Pattern/trend alignment</b> – how well the model captures the overall sentiment trends
      * `scipy.stats.pearsonr` (linear correlation)
      * `scipy.stats.spearmanr` (rank correlation)
### Results

