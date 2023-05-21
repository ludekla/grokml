## Grokking Machine Learning

# Book

Luis G. Serrano: Grokking Machine Learning, Manning Publications Co. (2021)

# Ch03 Linear Regression

/home/lutz/code/golang/grokml/data/Hyderabad.csv

# Ch05 Perceptron

/home/lutz/code/golang/grokml/data/returns.csv

# Ch06 Logistic Regression
 
/home/lutz/code/golang/grokml/data/IMDB_Dataset.csv
/home/lutz/code/golang/grokml/data/reviews.csv

TODO: 
get rid of DataSet struct in pkg/ch06-logreg in favour of generic DataSet
in utils/dataset.go like so

type Data interface {
	Vector | TokenMap
}

type DataSet[T Data] struct {
	...
	x []T
	y []float64
}

# Ch08 Naive Bayes

# Ch09 Decision Trees