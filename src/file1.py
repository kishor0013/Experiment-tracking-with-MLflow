import mlflow
import mlflow.sklearn 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

##To not get any errors while logging
mlflow.set_tracking_uri('http://127.0.0.1:5000')

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


max_depth = 5
n_estimators = 10  
# mlflow.autolog()  #to log all the things automatically 
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    #Creating a confusion matrix to log artifacts
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('COnfusion Matrix')

    plt.savefig('confusion_matrix.png')

    #Logging Confusion matrix and this file as artifacts
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    print(f"Accuracy :" ,accuracy)
    
    