"""
Based on:
https://www.kaggle.com/ankitakumar/linear-regression-using-wine-quality-dataset/notebook  
"""

import numpy as np
import pandas as pd
import pickle
import argparse 
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_csv_path', required=True)
    parser.add_argument('--output_metrics_path', required=True)
    parser.add_argument('--output_checkpoint_path', required=True)
 
    args = parser.parse_args()
    csv_path = args.input_csv_path
    metrics_path = args.output_metrics_path
    checkpoint_path = args.output_checkpoint_path
    
    df=pd.read_csv(csv_path, sep=";")

    X = df[list(df.columns)[:-1]]
    print(df.columns)
    y=df['quality']

    # TODO: seed
    X_train, X_test,y_train,y_test=train_test_split(X,y)
    
    regressor=LinearRegression()
    
    print("Training model...")
    regressor.fit(X_train,y_train)

    print("Evaluating metrics...")
    y_prediction=regressor.predict(X_test) 
    print('R-score is %s' % regressor.score(X_test,y_test))
    with open(metrics_path, 'w') as f:
        print('R-score is %s' % regressor.score(X_test,y_test), file=f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(regressor, f)

if __name__ == '__main__':
    main()    
