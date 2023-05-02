# data from Naive Bayes model
import matplotlib.pyplot as mpl

def get_data(csvfile):
    with open(csvfile) as cin:
        lines = [line.strip().split(',') for line in cin]
        x = [float(line[0]) for line in lines]
        y = [float(line[1]) for line in lines]
        return x, y

def auc(x, y):
    val = 0.0
    pairs = zip(x, y)
    x0, y0 = next(pairs) 
    for xi, yi in pairs:
        val += (yi + y0)*abs(xi - x0)/2.0
        x0, y0 = xi, yi
    return val

if __name__ == '__main__':

    x, y = get_data('roc.csv')
    mpl.plot(x, y)
    mpl.xlabel("Sensitivity")
    mpl.ylabel("Specificity")
    mpl.title("ROC - Receiver Operator Characteristic")
    mpl.show()
    a = auc(x, y)
    print(f'auc: {a}')