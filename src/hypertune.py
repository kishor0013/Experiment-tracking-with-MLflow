import pandas as pd 
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

import dagshub
dagshub.init(repo_owner='kishor0013', repo_name='Experiment-tracking-with-MLflow', mlflow=True)


##To not get any errors while logging
mlflow.set_tracking_uri('https://dagshub.com/kishor0013/Experiment-tracking-with-MLflow.mlflow')

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

params_grid ={
    'n_estimators' : [10,50,30],
    'max_depth' : [None,4,8,10]
}

grid_search = GridSearchCV(estimator=rf,param_grid=params_grid,cv=5,n_jobs=-1,verbose=2)

mlflow.set_experiment('Hypertuning')

with mlflow.start_run() as parent :
    grid_search.fit(X_train,y_train)

    #Logging all the child runs as well
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested = True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('Accuracy',grid_search.cv_results_['mean_test_score'][i])

    #Logging the best model from grid search but first logging dataset without autolog
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric('Accuracy',best_score) 

    #pandas dataset should be converted into mlflow format
    train_df=X_train.copy()
    train_df['target']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,'training')


    test_df=X_test.copy()
    test_df['target']=y_test

    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'testing')

    #log the model and arifacts 
    mlflow.log_artifact(__file__)#source code
    
    mlflow.sklearn.log_model(grid_search.best_estimator_,'Random Forest')

    mlflow.set_tags({'author':"Kishor"})

    print(grid_search.best_params_)
    print(grid_search.best_score_)

