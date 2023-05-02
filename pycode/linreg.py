import argparse
import sys
import yaml
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split 
# import pickle as pkl # alternative to joblib

from pathlib import Path
from joblib import load, dump

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--train', 
    action='store_true',
    help='run training'
)

def fetch_data(path, target_col, *feature_cols):
    tb = pd.read_csv(path)
    return tb[list(feature_cols)], tb[target_col]

def get_parms(filename):
    with open(filename) as ymlfile:
        return yaml.safe_load(ymlfile)


if __name__ == '__main__':

    args = parser.parse_args()

    parms = get_parms('lrparms.yml')

    model_path = Path('lr.joblib')
    
    if args.train:
        print('fitting a new model')
        # get data
        X, y = fetch_data(parms['path'], parms['target'], *parms['features'])
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1) 
        lr = lm.LinearRegression()
        lr.fit(Xtrain, ytrain)
        dump(lr, str(model_path))
        print('testset score:', lr.score(Xtest, ytest))
    elif model_path.exists():
        print('use existing model')
        lr = load(str(model_path))
    else:
        print('ERROR: Need to train a model. Try again.')
        sys.exit(0)

    feats = parms['predict']
    yp = lr.predict([feats])
    print(f'prediction: {feats} -> {yp[0]:.2f}')
    


