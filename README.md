# Offenseval
SemEval 2019 - Task 6 - Identifying and Categorizing Offensive Language in Social Media 

## Descrption
A system to classifiy a tweet as either offensive or not offensive (Sub-task A) and further classifies offensive tweets into categories (Sub-tasks B – C). Some sort of grid search approach is taken where multiple techniques for preprocessing, feature extraction and classification are implemented and combinations of them all are tried to achieve the best model for the given dataset.

## Subtasks
- Sub-task A - Offensive language identification;  [Offensive: OFF, Not Offensive: NOT]
- Sub-task B - Automatic categorization of offense types; [Targted: TIN, Untargeted: UNT]
- Sub-task C: Offense target identification. [Individual: IND, Group: GRP, Other: OTH]

## Implementation

### Preprocessing
Tokenization, Stopwords Removal, Lemmatizaion, Stemming

### Vectorization
TFIDF, Count, Word2Vec, GloVe, fastText

### Classification
KNN, Naïve Bayes, SVM, Decision Trees, Random Forest, Logistic Regression, MLP, Adaboost, Bagging
#### Deeplearning
LSTM, 1-D CNN

## Running
- Install requiremetns using `pip3 install -r requirements.txt`
 
- `python3 tune.py` to do a complete search on all combinations of prepocessing, vectorization and non-deep classification techniques while tuning the classifiers hyper-params.

- `python3 train.py` to train one of the deeplearning models.

## Sample tuning.py output
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/LogisticRegression.png?raw=true "Logistic Regression")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/SVC.png?raw=true "SVC")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-a/RandomForest.png?raw=true "RF")
