import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def get_data(path):
    return pd.read_csv(path, index_col=0)

if __name__ == '__main__':

    for file in ['linear', 'one_circle', 'two_circles']:
        print(file)
        path = f'manning/Chapter_11_Support_Vector_Machines/{file}.csv'
        data = get_data(path)

        X, y = data[['x_1', 'x_2']], data['y']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)

        for c in [0.01, 1, 10, 100]:
            # svm = SVC(kernel='linear', C=c)
            # svm = SVC(kernel='poly', degree=2, gamma='auto')
            svm = SVC(kernel='rbf', gamma=c)
            svm.fit(Xtrain, ytrain)
            s = svm.score(Xtest, ytest)
            print(f'C: {c}, Score: {s}')