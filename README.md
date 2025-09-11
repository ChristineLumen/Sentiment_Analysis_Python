# Sentiment_Analysis_VADER vs RoBERTa

### Project Overview
This project will perform sentiment analysis by comparing two models: VADER (a lexicon-based sentiment model) and RoBERTa (a pre-trained model for sentiment classification). The goal is to determine which model provides more accurate results and performs better.
### Data Source
The dataset used for this analysis was downloaded from Kaggle. It contains detailed information including but not limited to customer's age, price range, location and reviews.  
In total, it includes 15 columns and 7,394 rows.
<br/> 
Source: <a href="https://www.kaggle.com/datasets/adarsh0806/influencer-merchandise-sales">Kaggle, Merchandise Sales</a>

### Tools
**Python libraries:**  
- `matplotlib` – visualization  
- `numpy` – numerical computing  
- `seaborn` – statistical plots
- `tqdm` – progress bars

**Sentiment analysis models:**  
- **VADER**  
  - `nltk.sentiment.SentimentIntensityAnalyzer`  
- **RoBERTa**  
  - `transformers`
  - `scipy.special`
    
### Running Model
 - VADER<br>
  I ran the VADER model and plotted its results. First, I generated four polarity scores (negative, neutral, positive, compound) for each review. For this purpose, I used `nltk.sentiment.SentimentIntensityAnalyzer`. Then, I stored these scores in lists, added them as new columns to my DataFrame and finally exported the DataFrame to an Excel file. Lastly, I used `sns.barplot` and `plt.subplots` to visualize the results.
<img width="900" height="300" alt="Vader_sentiment" src="https://github.com/user-attachments/assets/fa64a44f-c411-4cc1-9a78-3829ddea7d5f" />

 - RoBERTa<br> 
   Next I run RoBERTa model as it offers deeper analysis and is better at capturing contexstual differences in review. So for RoBERTa I started with uploading a pretrained model `f'cardiffnlp/twitter-roberta-base-sentiment'` with `transformers.AutoModelForSequenceClassification` and applied `scipy.special.softmax` to get the probability scores.
   <img width="885" height="295" alt="Screenshot 2025-08-19 at 5 18 05 PM" src="https://github.com/user-attachments/assets/844f37c6-fb95-463d-81de-2ae1d3bebc26" />

### Accuracy
First, I rescaled the predictions to a 1–5 rating range for both models so they match the human ratings, since originally they were in the range of –1 to 1. For this, I used:<br>
      * `y_pred_vader_scaled = 2 * (df['vader_compound'] + 1)`  -- shifts the range and stretches it to [0, 4] <br>
      * `y_pred_vader_scaled = y_pred_vader_scaled + 1` -- shifts again to [1, 5]<br>
For measuring model accuracy, I used regression metrics (MSE, MAE, Pearson and Spearman correlations) to evaluate how close the model’s sentiment predictions are to human ratings:
* <b>Size of errors</b> – how far off the predictions are from actual ratings<br>
      * `sklearn.metrics.mean_squared_error`<br>
      * `sklearn.metrics.mean_absolute_error`
* <b>Pattern/trend alignment</b> – how well the model captures the overall sentiment trends<br>
      * `scipy.stats.pearsonr` (linear correlation)<br>
      * `scipy.stats.spearmanr` (rank correlation)<br>
<img width="318" height="98" alt="Screenshot 2025-08-19 at 10 48 37 PM" src="https://github.com/user-attachments/assets/0fd835f0-2513-4fd5-95d0-5ce6f0c21616" /><br>
As you can see, VADER is almost always off by about 1 rating point, which means it can make larger errors. In contrast, RoBERTa is closer to the human ratings and less prone to mistakes.

### Results
Based on the results from both models, I created a scatter plot comparing each model’s predictions with the human ratings to show how closely they align.<br>
* VADER: Spread out, less aligned 
* RoBERTa: Clustered, aligned with trends → tracks the ups/downs of human ratings better<br>
<img width="526" height="555" alt="Screenshot 2025-08-19 at 10 49 55 PM" src="https://github.com/user-attachments/assets/d8897744-8f69-4495-beb2-100de9507b03" /><br>
Based on the chart, we can see that VADER has a moderate correlation with human ratings, meaning it captures general trends but weak on nuanced context. On the other hand, RoBERTa has a stronger correlation with human rating both in terms of linear trend(Pearson) and rank ordering (Spearman).
 
* VADER is simplier and works well for predict sentiment polarity.
* RoBERTa is much more **accurate** and closely **matches** how **humans** perceive **sentiment**, both in predicting the score and in ranking positive and negative reviews
