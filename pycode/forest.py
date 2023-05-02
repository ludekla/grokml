import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_data(path):
    data = pd.read_csv(path, index_col=0)
    data['Admitted'] = data['Chance of Admit'] > 0.5
    data = data.drop(['Chance of Admit'], axis=1)
    features = data.drop(['Admitted'], axis=1)
    labels = data['Admitted']
    return features, labels

if __name__ == '__main__':

    path = "manning/Chapter_9_Decision_Trees/Admission_Predict.csv"
    
    print('Classification')
    features, labels = get_data(path)
    dt = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    print('Accuracy Decision Tree:', acc)

    rfc = RandomForestClassifier(n_estimators=5, max_depth=1)
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    print('Accuracy Random Forest:', acc)

    feats = [[10], [20], [30], [40], [50], [60], [70], [80], [86]]
    targs = [7, 5, 7, 1, 2, 1, 5, 4, 3.6]
    treg = DecisionTreeRegressor()
    treg.fit(feats, targs)
    score = treg.score(feats, targs)
    print('Regression score:', score)

    print(treg.predict(feats))
