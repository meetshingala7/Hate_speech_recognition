# Hate_speech_recognition
[dataset card on huggingface](https://huggingface.co/datasets/hate_speech_offensive)

We have created a model to classify text into 3 different classes: 
   0 - hate speech 
   1 - offensive language
   2 - neither


For preprocessing of text we used regex to remove all the urls in any tweets that may have them and replaced them with blankspaces.
```
df['tweet'] = df['tweet'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
```

We then took the text classification using TFIDF vectorizer to produce vectors and tokens for classification.

Models used:
1. *Multinomial Naive Bayes (sklearn.MultinomialNB)*
   1. **Without GridSearchCV**
      For all the default parameters.
      We ran it for 40 differents training splits from 50 to 90% as the training split
      and we concluded that for the data, the best training split is at 86% precentage of training data, predicting on the testing data giving an accuracy of 80.979%.
   3. **With GridsearchCV** (CV=5,default)
      The paramters passed were for different learning rates at 86% training split.
      ```python
      alpha = [1,0.5,0.1,0.01,0.001,0.0001]
      param = {'alpha': alpha}
      ```
      We got the maximum accuracy at 83.717% for alpha = 0.1
2. *Support Vector Classifier (sklearn.SVM.SVC)*
   1. **With GridsearchCV** (CV=5,default)
      We ran it for the following paramters with gridsearchcv
      ```python
      {
         'C': [0.1,1,10,100,1000],
         'gamma': [1,0.1,0.01,0.001,0.0001],
         'kernel': ['rbf','linear','poly'],
      }
      ```
      We got the maximum accuracy at 86% training split with
3. *K-Nearest Neighbours (sklearn.neighbors.KNN)*
   1. **Without GridsearchCV**
      We ran it for 74 unique neigbors ranging from 1 to 150 with all odd numbers.
      We received the highest accuracy for k=7 neighbors.
   2. **With GridSearchCV** (CV=3,custom)
      We received the maximum accuracy of 82.947%
4. *Multilayer Perceptron Model (sklearn.neural_network.MLPClassifier)*
   1. **Without GridSearchCV**
      We ran the model to get the best train split which we got at 65% training data and 35% testing splits
      We received the maximum accuracy of 89.428% for the same.
   2. **With GridSearchCV** (CV=3,custom)
      We ran the model for the following parameters.
      ```python
      params = {
            'solver': ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes':[(4), (5,),(6,), (7,)],
            'activation':['identity', 'logistic', 'tanh', 'relu']
      }
      ```
      The best accuracy at 88.107% came at the following parameter (at 85% training split Without the gridsearchCV as it was with 65% training split)
      ```python
      {'activation': 'logistic', 'hidden_layer_sizes': (6,), 'solver': 'lbfgs'}
      ```
5. *Decision Tree Classifier (sklearn.tree.DecisionTreeClassifier)*
   1. **Without GridSearchCV**
      We ran the model for gini and entropy criterions.The model accuracy tops at 90%(highest split) was 78.45% and 78.43% respectively

   2. **With GridSearchCV** (CV=100,custom)
      We ran the models for the following parameters.
      with both 'gini' and 'entropy'
      and the best accuracy was at 89.28 with 'entropy' criterion 






   
