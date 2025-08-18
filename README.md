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
 - VADER<br>
  I ran the VADER model and plotted its results. First, I generated four polarity scores (negative, neutral, positive, compound) for each review. For this purpose, I used `nltk.sentiment.SentimentIntensityAnalyzer`. Then, I stored these scores in lists, added them as new columns to my DataFrame and finally exported the DataFrame to an Excel file. Lastly, I used `sns.barplot` and `plt.subplots` to visualize the results.
<img width="900" height="300" alt="Vader_sentiment" src="https://github.com/user-attachments/assets/fa64a44f-c411-4cc1-9a78-3829ddea7d5f" />

 - RoBERTa 
   
### Accuracy
For measuring model accuracy, I used regression metrics (MSE, MAE, Pearson and Spearman correlations) to evaluate how close the model’s sentiment predictions are to human ratings:
* <b>Size of errors</b> – how far off the predictions are from actual ratings<br>
      * `sklearn.metrics.mean_squared_error`<br>
      * `sklearn.metrics.mean_absolute_error`
* <b>Pattern/trend alignment</b> – how well the model captures the overall sentiment trends<br>
      * `scipy.stats.pearsonr` (linear correlation)<br>
      * `scipy.stats.spearmanr` (rank correlation)<br>
  <img width="495" height="100" alt="Screenshot 2025-08-17 at 11 12 41 PM" src="https://github.com/user-attachments/assets/4146d370-b448-414c-aaf4-0e1f4a1ddd10" />

### Results

