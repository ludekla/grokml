import argparse
import joblib
import sys
import yaml
from pathlib import Path

import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split 
# import pickle as pkl # alternative to joblib

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--train', 
    action='store_true',
    help='run training'
)

def fetch_data(path, target_col, *feature_cols):
    tb = pd.read_csv(path)
    return tb[list(feature_cols)], tb[target_col]

if __name__ == '__main__':

    args = parser.parse_args()

    model_path = Path('config/lr.joblib')
    
    if args.train:
        print('fitting a new model')
        # get data
        path = 'data/Hyderabad.csv'
        X, y = fetch_data(path, 'Price', 'Area', 'No. of Bedrooms')
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1) 
        lr = lm.LinearRegression()
        lr.fit(Xtrain, ytrain)
        joblib.dump(lr, str(model_path))
        print('testset score:', lr.score(Xtest, ytest))
    elif model_path.exists():
        print('use existing model')
        lr = joblib.load(str(model_path))
    else:
        print('ERROR: Need to train a model. Try again.')
        sys.exit(0)

    examples = [[600, 1], [1000, 2], [1500, 3], [2000, 4]]
    yp = lr.predict(examples)
    for ex, y in zip(examples, yp):
        print(f'prediction: {ex} -> {y:.2f}')
    


