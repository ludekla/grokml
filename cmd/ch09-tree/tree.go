package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch09-tree"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	ds := ch09.NewDataSet("data/Admission_Predict.csv")
	trainSet, testSet := ds.Split(0.1)

	var dt1, dt2 ch09.TreeClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		dt1 = ch09.NewTreeClassifier(ch09.NewEntropy(0.5), 0.1)
		dt1.Fit(trainSet)
		dt1.Save("models/tree_entropy.json")
		// Gini
		dt2 = ch09.NewTreeClassifier(ch09.NewGini(0.5), 0.1)
		dt2.Fit(trainSet)
		dt2.Save("models/tree_gini.json")
	} else {
		// Entropy
		dt1 = ch09.TreeClassifier{}
		dt1.Load("models/tree_entropy.json")
		// Gini
		dt2 = ch09.TreeClassifier{}
		dt2.Load("models/tree_gini.json")
	}
	// Scoring tests
	rep := dt1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = dt2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
