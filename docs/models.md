
# Models

I made this package mostly to get my hands dirty with sklearn, i ended up 
testing lots of models

There are better tools and ways to find the best model, this was just a learning experience

- [Models](#models)
  * [Features](#features)
  * [Label Classification](#label-classification)
    + [Default Pipeline](#default-pipeline)
    + [Logistic Regression](#logistic-regression)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier)
      - [Main Label Question Classifier](#main-label-question-classifier)
    + [SVM](#svm)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-1)
      - [Main Label Question Classifier](#main-label-question-classifier-1)
      - [Sentence Classifier](#sentence-classifier)
    + [Decision Tree](#decision-tree)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-2)
      - [Main Label Question Classifier](#main-label-question-classifier-2)
      - [Sentence Classifier](#sentence-classifier-1)
    + [SGD](#sgd)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-3)
      - [Main Label Question Classifier](#main-label-question-classifier-3)
      - [Sentence Classifier](#sentence-classifier-2)
    + [Ridge](#ridge)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-4)
      - [Main Label Question Classifier](#main-label-question-classifier-4)
      - [Sentence Classifier](#sentence-classifier-3)
    + [Perceptron](#perceptron)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-5)
      - [Main Label Question Classifier](#main-label-question-classifier-5)
      - [Sentence Classifier](#sentence-classifier-4)
    + [Passive Aggressive](#passive-aggressive)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-6)
      - [Main Label Question Classifier](#main-label-question-classifier-6)
      - [Sentence Classifier](#sentence-classifier-5)
    + [Naive Bayes](#naive-bayes)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-7)
      - [Main Label Question Classifier](#main-label-question-classifier-7)
    + [Gradient Boosting](#gradient-boosting)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-8)
      - [Main Label Question Classifier](#main-label-question-classifier-8)
    + [Random Forest](#random-forest)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-9)
      - [Main Label Question Classifier](#main-label-question-classifier-9)
    + [AdaBoost](#adaboost)
      - [Main:Secondary Label Question Classifier](#main-secondary-label-question-classifier-10)
      - [Main Label Question Classifier](#main-label-question-classifier-10)
      - [Sentence Classifier](#sentence-classifier-6)


## Features

For features i played around with

- CountVectorizer, n_gram range (1,2,3), lemmatized input (True, False)
- TfidfVectorizer, n_gram range (1,2,3), lemmatized input (True, False)
- Word2Vec, lemmatized input (True, False)
- PosTagVectorizer
- IntentDataVectorizer (Basic, Padaos, Padatious)


I then mixed these and did an extensive search over different pipelines

All models were trained on a single feature or a FeatureUnion of combinations


## Label Classification

I trained 3 classifiers, Main Label, Main + Secondary label, Command or Question

Main + Secondary label pairs were considered a single label, future work 
will classify the secondary label after the first label

Data for sentence classification was very limited, in the package i ended up
 using heuristics based on eye-balled features
 
 
### Default Pipeline

in order to have a baseline i trained all models with a default pipeline as 
follows

- CountVectorizer, n_gram range (1,2)
- TfidfVectorizer, n_gram range (1,2), lemmatized input
- Word2Vec
- PosTagVectorizer

No hyperparameter optimization has been done yet

Bellow you have a report for each model for the default pipeline


### Logistic Regression

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.794

                       precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.75      0.86         8
          DESC:def       0.79      1.00      0.88       123
         DESC:desc       0.33      0.71      0.45         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.70      0.44      0.54        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.33      0.50         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.25      0.40         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.33      0.33      0.33        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.50      0.86      0.63         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.84      0.96      0.90        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.78      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.76      0.88      0.81        50
         LOC:state       0.62      0.71      0.67         7
         NUM:count       0.90      1.00      0.95         9
          NUM:date       0.98      0.96      0.97        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       0.50      0.33      0.40         3
         NUM:other       0.86      0.50      0.63        12
          NUM:perc       0.67      0.67      0.67         3
        NUM:period       0.73      1.00      0.84         8
         NUM:speed       1.00      0.50      0.67         6
          NUM:temp       1.00      0.20      0.33         5
        NUM:weight       1.00      0.25      0.40         4
    
         micro avg       0.79      0.79      0.79       500
         macro avg       0.72      0.61      0.62       500
      weighted avg       0.81      0.79      0.77       500

#### Main Label Question Classifier

default pipeline - Accuracy: 0.894

                       precision    recall  f1-score   support
        
                ABBR       1.00      0.78      0.88         9
                DESC       0.84      0.99      0.91       138
                ENTY       0.84      0.80      0.82        94
                 HUM       0.92      0.92      0.92        65
                 LOC       0.90      0.86      0.88        81
                 NUM       1.00      0.88      0.93       113
        
           micro avg       0.89      0.89      0.89       500
           macro avg       0.92      0.87      0.89       500
        weighted avg       0.90      0.89      0.89       500
 
     

### SVM

#### Main:Secondary Label Question Classifier

- lemmatized text, count vectorizer ngram(1,2) - Accuracy: 0.836
- lemmatized text, tfidf vectorizer ngram(1,2) - Accuracy: 0.822
- postag one hot encoder - Accuracy: 0.802
- default pipeline - Accuracy: 0.838

best pipeline so far: cv2_lemma_w2v_lemma_ner Accuracy: 0.844
    
                     precision    recall  f1-score   support

          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.88      0.93         8
          DESC:def       0.83      1.00      0.91       123
         DESC:desc       0.56      0.71      0.63         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.83      0.62      0.71        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.83      0.91         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.60      0.75      0.67         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.38      0.42      0.40        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       0.50      1.00      0.67         1
       ENTY:termeq       0.55      0.86      0.67         7
          ENTY:veh       1.00      0.50      0.67         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.95      0.98      0.96        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.78      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.76      0.84      0.80        50
         LOC:state       0.70      1.00      0.82         7
         NUM:count       0.90      1.00      0.95         9
          NUM:date       1.00      1.00      1.00        47
          NUM:dist       0.90      0.56      0.69        16
         NUM:money       1.00      0.33      0.50         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.50      0.67      0.57         3
        NUM:period       0.67      1.00      0.80         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.80      1.00      0.89         4

         micro avg       0.84      0.84      0.84       500
         macro avg       0.75      0.70      0.69       500
      weighted avg       0.85      0.84      0.82       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.902

                     precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.84      0.99      0.91       138
            ENTY       0.85      0.80      0.82        94
             HUM       0.94      0.95      0.95        65
             LOC       0.92      0.85      0.88        81
             NUM       0.99      0.90      0.94       113
    
       micro avg       0.90      0.90      0.90       500
       macro avg       0.92      0.88      0.90       500
    weighted avg       0.91      0.90      0.90       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.8666666666666667


                     precision    recall  f1-score   support
    
         command       1.00      0.80      0.89         5
        question       0.80      0.80      0.80         5
       statement       0.83      1.00      0.91         5
    
       micro avg       0.87      0.87      0.87        15
       macro avg       0.88      0.87      0.87        15
    weighted avg       0.88      0.87      0.87        15
    
### Decision Tree

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.666
    
                  precision    recall  f1-score   support
    
          ABBR:abb       0.50      1.00      0.67         1
          ABBR:exp       0.86      0.75      0.80         8
          DESC:def       0.90      0.92      0.91       123
         DESC:desc       0.21      0.43      0.29         7
       DESC:manner       0.50      1.00      0.67         2
       DESC:reason       0.71      0.83      0.77         6
       ENTY:animal       0.21      0.19      0.20        16
         ENTY:body       0.50      0.50      0.50         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.00      0.00      0.00         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.29      0.50      0.36         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      0.50      0.67         2
       ENTY:letter       0.00      0.00      0.00         0
        ENTY:other       0.19      0.25      0.21        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       0.75      0.20      0.32        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.11      0.14      0.12         7
          ENTY:veh       0.00      0.00      0.00         4
         ENTY:word       0.00      0.00      0.00         0
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.17      0.33      0.22         6
           HUM:ind       0.77      0.91      0.83        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.67      0.80        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.33      0.33      0.33         3
         LOC:other       0.71      0.60      0.65        50
         LOC:state       0.60      0.43      0.50         7
          NUM:code       0.00      0.00      0.00         0
         NUM:count       0.80      0.89      0.84         9
          NUM:date       0.93      0.91      0.92        47
          NUM:dist       0.70      0.44      0.54        16
         NUM:money       0.33      0.33      0.33         3
         NUM:other       0.50      0.42      0.45        12
          NUM:perc       0.33      0.33      0.33         3
        NUM:period       0.60      0.75      0.67         8
         NUM:speed       1.00      0.17      0.29         6
          NUM:temp       0.00      0.00      0.00         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       0.50      0.25      0.33         4
    
         micro avg       0.67      0.67      0.67       500
         macro avg       0.47      0.44      0.44       500
      weighted avg       0.70      0.67      0.67       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.784

                  precision    recall  f1-score   support
    
            ABBR       0.73      0.89      0.80         9
            DESC       0.84      0.92      0.88       138
            ENTY       0.65      0.56      0.61        94
             HUM       0.68      0.83      0.75        65
             LOC       0.80      0.73      0.76        81
             NUM       0.88      0.81      0.84       113
    
       micro avg       0.78      0.78      0.78       500
       macro avg       0.76      0.79      0.77       500
    weighted avg       0.78      0.78      0.78       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.8666666666666667


                  precision    recall  f1-score   support
    
         command       1.00      0.80      0.89         5
        question       1.00      0.80      0.89         5
       statement       0.71      1.00      0.83         5
    
       micro avg       0.87      0.87      0.87        15
       macro avg       0.90      0.87      0.87        15
    weighted avg       0.90      0.87      0.87        15
    
### SGD

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.802
    
                     precision    recall  f1-score   support
    
          ABBR:abb       0.20      1.00      0.33         1
          ABBR:exp       0.30      0.88      0.45         8
          DESC:def       0.97      0.98      0.98       123
         DESC:desc       0.46      0.86      0.60         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.86      0.38      0.52        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       0.83      1.00      0.91        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.67      0.80         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.60      0.75      0.67         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.50      0.33      0.40        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.50      0.25      0.33         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       0.86      0.40      0.55        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.47      1.00      0.64         7
          ENTY:veh       0.67      0.50      0.57         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       1.00      0.33      0.50         6
           HUM:ind       0.88      0.95      0.91        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.79      0.83      0.81        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.85      0.70      0.77        50
         LOC:state       0.58      1.00      0.74         7
         NUM:count       1.00      0.89      0.94         9
          NUM:date       0.98      1.00      0.99        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       0.17      0.67      0.27         3
           NUM:ord       0.00      0.00      0.00         0
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       1.00      0.67      0.80         3
        NUM:period       0.62      1.00      0.76         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.60      0.75         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       1.00      0.50      0.67         4
    
         micro avg       0.80      0.80      0.80       500
         macro avg       0.69      0.64      0.62       500
      weighted avg       0.86      0.80      0.80       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.888

                     precision    recall  f1-score   support
    
            ABBR       0.58      0.78      0.67         9
            DESC       0.88      0.99      0.93       138
            ENTY       0.88      0.67      0.76        94
             HUM       0.92      0.94      0.93        65
             LOC       0.82      0.91      0.87        81
             NUM       0.98      0.91      0.94       113
    
       micro avg       0.89      0.89      0.89       500
       macro avg       0.84      0.87      0.85       500
    weighted avg       0.89      0.89      0.89       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.5333333333333333


                   precision    recall  f1-score   support
    
         command       0.00      0.00      0.00         5
        question       0.67      0.80      0.73         5
       statement       0.44      0.80      0.57         5
    
       micro avg       0.53      0.53      0.53        15
       macro avg       0.37      0.53      0.43        15
    weighted avg       0.37      0.53      0.43        15
    
    
### Ridge

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.834
    
                        precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.62      0.77         8
          DESC:def       0.77      1.00      0.87       123
         DESC:desc       1.00      0.71      0.83         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.91      0.62      0.74        16
         ENTY:body       0.67      1.00      0.80         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.83      0.83      0.83         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.75      0.86         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.45      0.42      0.43        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       1.00      0.25      0.40         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       1.00      0.40      0.57        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.67      0.86      0.75         7
          ENTY:veh       1.00      0.50      0.67         4
          HUM:desc       1.00      0.67      0.80         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.90      0.96      0.93        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.93      0.78      0.85        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.78      0.80      0.79        50
         LOC:state       0.58      1.00      0.74         7
         NUM:count       0.82      1.00      0.90         9
          NUM:date       1.00      0.98      0.99        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       1.00      0.33      0.50         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.60      1.00      0.75         3
        NUM:period       0.80      1.00      0.89         8
         NUM:speed       1.00      0.83      0.91         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.80      1.00      0.89         4
    
         micro avg       0.83      0.83      0.83       500
         macro avg       0.81      0.71      0.73       500
      weighted avg       0.85      0.83      0.82       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.896

                 precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.82      0.99      0.90       138
            ENTY       0.87      0.76      0.81        94
             HUM       0.94      0.94      0.94        65
             LOC       0.90      0.88      0.89        81
             NUM       1.00      0.90      0.95       113
    
       micro avg       0.90      0.90      0.90       500
       macro avg       0.92      0.87      0.89       500
    weighted avg       0.90      0.90      0.90       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.6666666666666666


                  precision    recall  f1-score   support
    
         command       1.00      0.20      0.33         5
        question       0.80      0.80      0.80         5
       statement       0.56      1.00      0.71         5
    
       micro avg       0.67      0.67      0.67        15
       macro avg       0.79      0.67      0.62        15
    weighted avg       0.79      0.67      0.62        15
    
### Perceptron

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.766
    
                    precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.62      0.77         8
          DESC:def       0.71      1.00      0.83       123
         DESC:desc       0.71      0.71      0.71         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.73      0.50      0.59        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.67      0.33      0.44         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.00      0.00      0.00         4
       ENTY:instru       0.50      1.00      0.67         1
         ENTY:lang       1.00      0.50      0.67         2
        ENTY:other       0.50      0.25      0.33        12
        ENTY:plant       1.00      0.40      0.57         5
      ENTY:product       0.00      0.00      0.00         4
     ENTY:religion       0.00      0.00      0.00         0
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       0.80      0.27      0.40        15
       ENTY:symbol       0.00      0.00      0.00         0
     ENTY:techmeth       0.25      1.00      0.40         1
       ENTY:termeq       0.50      0.71      0.59         7
          ENTY:veh       0.67      0.50      0.57         4
         ENTY:word       0.00      0.00      0.00         0
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.33      0.67      0.44         6
           HUM:ind       1.00      0.91      0.95        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.82      0.78      0.80        18
       LOC:country       0.75      1.00      0.86         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.86      0.76      0.81        50
         LOC:state       0.67      0.57      0.62         7
         NUM:count       1.00      1.00      1.00         9
          NUM:date       0.92      0.96      0.94        47
          NUM:dist       0.89      0.50      0.64        16
         NUM:money       0.33      0.33      0.33         3
         NUM:other       0.86      0.50      0.63        12
          NUM:perc       1.00      0.33      0.50         3
        NUM:period       0.62      1.00      0.76         8
         NUM:speed       0.80      0.67      0.73         6
          NUM:temp       1.00      0.20      0.33         5
        NUM:weight       0.00      0.00      0.00         4
    
         micro avg       0.77      0.77      0.77       500
         macro avg       0.62      0.54      0.54       500
      weighted avg       0.78      0.77      0.75       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.872

                 precision    recall  f1-score   support
    
            ABBR       0.80      0.89      0.84         9
            DESC       0.92      0.97      0.94       138
            ENTY       0.88      0.56      0.69        94
             HUM       0.81      0.95      0.87        65
             LOC       0.82      0.90      0.86        81
             NUM       0.90      0.94      0.92       113
    
       micro avg       0.87      0.87      0.87       500
       macro avg       0.85      0.87      0.85       500
    weighted avg       0.87      0.87      0.87       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.7333333333333333

                  precision    recall  f1-score   support
    
         command       0.75      0.60      0.67         5
        question       0.67      0.80      0.73         5
       statement       0.80      0.80      0.80         5
    
       micro avg       0.73      0.73      0.73        15
       macro avg       0.74      0.73      0.73        15
    weighted avg       0.74      0.73      0.73        15
    
### Passive Aggressive

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.804
    
                   precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       0.86      0.75      0.80         8
          DESC:def       0.81      0.99      0.89       123
         DESC:desc       0.45      0.71      0.56         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.67      0.38      0.48        16
         ENTY:body       0.67      1.00      0.80         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.33      0.50         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.75      0.75      0.75         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
       ENTY:letter       0.00      0.00      0.00         0
        ENTY:other       0.42      0.42      0.42        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       0.20      1.00      0.33         1
       ENTY:termeq       0.54      1.00      0.70         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.75      0.50      0.60         6
           HUM:ind       0.93      0.95      0.94        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.93      0.78      0.85        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.77      0.82      0.80        50
         LOC:state       0.60      0.86      0.71         7
         NUM:count       0.82      1.00      0.90         9
          NUM:date       1.00      0.98      0.99        47
          NUM:dist       0.90      0.56      0.69        16
         NUM:money       0.50      0.33      0.40         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.50      0.67      0.57         3
        NUM:period       0.67      1.00      0.80         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.75      0.75      0.75         4
    
         micro avg       0.80      0.80      0.80       500
         macro avg       0.70      0.65      0.64       500
      weighted avg       0.83      0.80      0.79       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.882

                 precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.82      0.99      0.89       138
            ENTY       0.84      0.72      0.78        94
             HUM       0.90      0.95      0.93        65
             LOC       0.90      0.85      0.87        81
             NUM       0.99      0.88      0.93       113
    
       micro avg       0.88      0.88      0.88       500
       macro avg       0.91      0.86      0.88       500
    weighted avg       0.89      0.88      0.88       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.8666666666666667

                  precision    recall  f1-score   support
    
         command       1.00      0.80      0.89         5
        question       0.80      0.80      0.80         5
       statement       0.83      1.00      0.91         5
    
       micro avg       0.87      0.87      0.87        15
       macro avg       0.88      0.87      0.87        15
    weighted avg       0.88      0.87      0.87        15
    
### Naive Bayes

NOTE: word2vec not used in this pipeline

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.53
    
                            precision    recall  f1-score   support
        
              ABBR:abb       0.00      0.00      0.00         1
              ABBR:exp       0.00      0.00      0.00         8
              DESC:def       0.80      1.00      0.89       123
             DESC:desc       0.75      0.43      0.55         7
           DESC:manner       0.67      1.00      0.80         2
           DESC:reason       1.00      0.17      0.29         6
           ENTY:animal       0.00      0.00      0.00        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       0.00      0.00      0.00        10
         ENTY:currency       0.00      0.00      0.00         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       0.00      0.00      0.00         1
             ENTY:lang       0.00      0.00      0.00         2
            ENTY:other       0.00      0.00      0.00        12
            ENTY:plant       0.00      0.00      0.00         5
          ENTY:product       0.00      0.00      0.00         4
            ENTY:sport       0.00      0.00      0.00         1
        ENTY:substance       0.00      0.00      0.00        15
         ENTY:techmeth       0.00      0.00      0.00         1
           ENTY:termeq       0.00      0.00      0.00         7
              ENTY:veh       0.00      0.00      0.00         4
              HUM:desc       0.00      0.00      0.00         3
                HUM:gr       0.00      0.00      0.00         6
               HUM:ind       0.25      1.00      0.41        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       1.00      0.06      0.11        18
           LOC:country       1.00      0.33      0.50         3
             LOC:mount       0.00      0.00      0.00         3
             LOC:other       0.61      0.74      0.67        50
             LOC:state       0.00      0.00      0.00         7
             NUM:count       0.33      1.00      0.50         9
              NUM:date       1.00      0.68      0.81        47
              NUM:dist       0.00      0.00      0.00        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       0.00      0.00      0.00        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       1.00      0.12      0.22         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.53      0.53      0.53       500
             macro avg       0.20      0.16      0.14       500
          weighted avg       0.47      0.53      0.44       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.81

                     precision    recall  f1-score   support
        
                ABBR       0.00      0.00      0.00         9
                DESC       0.81      0.98      0.89       138
                ENTY       0.63      0.76      0.69        94
                 HUM       0.84      0.89      0.87        65
                 LOC       0.86      0.79      0.83        81
                 NUM       0.99      0.68      0.81       113
        
           micro avg       0.81      0.81      0.81       500
           macro avg       0.69      0.68      0.68       500
        weighted avg       0.82      0.81      0.80       500
    
    
### Gradient Boosting

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.776
    
                    precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.75      0.86         8
          DESC:def       0.88      0.98      0.93       123
         DESC:desc       0.80      0.57      0.67         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       1.00      0.50      0.67        16
         ENTY:body       0.00      0.00      0.00         2
        ENTY:color       1.00      1.00      1.00        10
     ENTY:currency       0.83      0.83      0.83         6
       ENTY:dismed       0.50      0.50      0.50         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.25      0.40         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       0.00      0.00      0.00         2
        ENTY:other       0.16      0.67      0.26        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       0.83      0.33      0.48        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.83      0.71      0.77         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.33      0.17      0.22         6
           HUM:ind       0.67      0.95      0.78        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.94      0.83      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.84      0.74      0.79        50
         LOC:state       0.67      0.86      0.75         7
         NUM:count       0.69      1.00      0.82         9
          NUM:date       1.00      0.96      0.98        47
          NUM:dist       0.89      0.50      0.64        16
         NUM:money       0.00      0.00      0.00         3
         NUM:other       1.00      0.08      0.15        12
          NUM:perc       1.00      0.67      0.80         3
        NUM:period       0.75      0.75      0.75         8
         NUM:speed       1.00      0.83      0.91         6
          NUM:temp       1.00      0.60      0.75         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       0.80      1.00      0.89         4
    
         micro avg       0.78      0.78      0.78       500
         macro avg       0.70      0.60      0.61       500
      weighted avg       0.81      0.78      0.76       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.858

                  precision    recall  f1-score   support
    
            ABBR       0.89      0.89      0.89         9
            DESC       0.90      0.96      0.93       138
            ENTY       0.67      0.79      0.72        94
             HUM       0.88      0.88      0.88        65
             LOC       0.86      0.77      0.81        81
             NUM       1.00      0.85      0.92       113
    
       micro avg       0.86      0.86      0.86       500
       macro avg       0.87      0.85      0.86       500
    weighted avg       0.87      0.86      0.86       500
    

### Random Forest

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.636
    
                           precision    recall  f1-score   support
        
              ABBR:abb       1.00      1.00      1.00         1
              ABBR:exp       1.00      1.00      1.00         8
              DESC:def       0.65      1.00      0.79       123
             DESC:desc       0.18      0.43      0.25         7
           DESC:manner       0.11      1.00      0.20         2
           DESC:reason       0.80      0.67      0.73         6
           ENTY:animal       1.00      0.06      0.12        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       1.00      0.70      0.82        10
         ENTY:currency       1.00      0.33      0.50         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       0.00      0.00      0.00         1
             ENTY:lang       0.00      0.00      0.00         2
            ENTY:other       0.33      0.17      0.22        12
            ENTY:plant       0.00      0.00      0.00         5
          ENTY:product       0.00      0.00      0.00         4
            ENTY:sport       0.00      0.00      0.00         1
        ENTY:substance       0.00      0.00      0.00        15
         ENTY:techmeth       1.00      1.00      1.00         1
           ENTY:termeq       0.33      0.14      0.20         7
              ENTY:veh       0.00      0.00      0.00         4
              HUM:desc       1.00      0.67      0.80         3
                HUM:gr       0.00      0.00      0.00         6
               HUM:ind       0.50      0.95      0.66        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       1.00      0.50      0.67        18
           LOC:country       0.50      0.67      0.57         3
             LOC:mount       1.00      0.33      0.50         3
             LOC:other       0.67      0.72      0.69        50
             LOC:state       0.75      0.43      0.55         7
             NUM:count       0.75      1.00      0.86         9
              NUM:date       1.00      0.87      0.93        47
              NUM:dist       0.50      0.06      0.11        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       1.00      0.42      0.59        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       0.67      0.25      0.36         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.64      0.64      0.64       500
             macro avg       0.42      0.34      0.34       500
          weighted avg       0.61      0.64      0.57       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.798

                   precision    recall  f1-score   support
    
            ABBR       1.00      0.89      0.94         9
            DESC       0.79      0.99      0.87       138
            ENTY       0.62      0.71      0.66        94
             HUM       0.81      0.86      0.84        65
             LOC       0.84      0.67      0.74        81
             NUM       1.00      0.69      0.82       113
    
       micro avg       0.80      0.80      0.80       500
       macro avg       0.84      0.80      0.81       500
    weighted avg       0.82      0.80      0.80       500
    

### AdaBoost

#### Main:Secondary Label Question Classifier

default pipeline - Accuracy: 0.22
    
                        precision    recall  f1-score   support
        
              ABBR:abb       0.00      0.00      0.00         1
              ABBR:exp       0.00      0.00      0.00         8
              DESC:def       0.00      0.00      0.00       123
             DESC:desc       0.00      0.00      0.00         7
           DESC:manner       0.00      0.00      0.00         2
           DESC:reason       0.00      0.00      0.00         6
           ENTY:animal       0.00      0.00      0.00        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       1.00      0.80      0.89        10
         ENTY:currency       0.00      0.00      0.00         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       1.00      1.00      1.00         1
             ENTY:lang       0.00      0.00      0.00         2
            ENTY:other       0.00      0.00      0.00        12
            ENTY:plant       0.00      0.00      0.00         5
          ENTY:product       0.00      0.00      0.00         4
            ENTY:sport       0.00      0.00      0.00         1
        ENTY:substance       0.00      0.00      0.00        15
         ENTY:techmeth       0.00      0.00      0.00         1
           ENTY:termeq       0.00      0.00      0.00         7
              ENTY:veh       0.00      0.00      0.00         4
              HUM:desc       0.00      0.00      0.00         3
                HUM:gr       0.00      0.00      0.00         6
               HUM:ind       0.94      0.80      0.86        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       0.00      0.00      0.00        18
           LOC:country       0.00      0.00      0.00         3
             LOC:mount       0.00      0.00      0.00         3
             LOC:other       0.11      1.00      0.21        50
             LOC:state       0.00      0.00      0.00         7
             NUM:count       1.00      0.78      0.88         9
              NUM:date       0.00      0.00      0.00        47
              NUM:dist       0.00      0.00      0.00        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       0.00      0.00      0.00        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       0.00      0.00      0.00         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.22      0.22      0.22       500
             macro avg       0.10      0.10      0.09       500
          weighted avg       0.15      0.22      0.15       500
      
      
#### Main Label Question Classifier

default pipeline - Accuracy: 0.592

                      precision    recall  f1-score   support
        
                ABBR       1.00      0.78      0.88         9
                DESC       0.65      0.72      0.68       138
                ENTY       0.36      0.76      0.49        94
                 HUM       0.75      0.68      0.71        65
                 LOC       0.76      0.32      0.45        81
                 NUM       0.98      0.42      0.59       113
        
           micro avg       0.59      0.59      0.59       500
           macro avg       0.75      0.61      0.63       500
        weighted avg       0.71      0.59      0.60       500
    
#### Sentence Classifier

default pipeline - Accuracy: 0.4666666666666667

                         precision    recall  f1-score   support

             command       0.00      0.00      0.00         5
            question       0.67      0.40      0.50         5
           statement       0.42      1.00      0.59         5
        
           micro avg       0.47      0.47      0.47        15
           macro avg       0.36      0.47      0.36        15
        weighted avg       0.36      0.47      0.36        15
    