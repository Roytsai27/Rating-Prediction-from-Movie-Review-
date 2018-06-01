# Rating-Prediction-from-Movie-Review

### The purpose of this project is to use user's review to predict user's rating for movies.

## Overview
* Train and test data
* Text preprocessing 
* Vectorize the text data
* Dimension Reduction
* Modeling
* Parameter tuning

## Text Preprocessing & Vectorize
First we build a tokenizer for the Vectorizer.

Then we implement Stemming for the text preprocessing using nltk and combining it with sklearn CountVectorize Model

## Dimension Reduction 
Here we use TruncatedSVD to do the dimension reduction

## Modeling
Which model to use depends on your problem and your goal.
if you treat it as a classifcation problem and you aim to maximize the accuracy , you need to use a clasifier.
if you problem is a regression problem , then you would like to minimize Mse and a regression model would be good. 
### Classification  Model
* Logistic Regression
* SVM
* Random Forest
* Xgboost
* SGDClassifier
### Regression  Model
* Linear Regression
* Lasso Regression
* Random Forest Regressor
* SVR
#### Here we want to minimize MSE and we found that the best model for this data is SVR
## Parameter tuning
*  use_idf
*  n_components
*  ngram_range
*  gamma
 