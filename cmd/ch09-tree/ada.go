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

	var ac1, ac2 *ch09.AdaBoostClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		ac1 = ch09.NewAdaBoostClassifier(3, ch09.NewEntropy(0.5), 0.1)
		ac1.Fit(trainSet)
		ac1.Save("models/adaBoost_entropy.json")
		// Gini
		ac2 = ch09.NewAdaBoostClassifier(3, ch09.NewGini(0.5), 0.1)
		ac2.Fit(trainSet)
		ac2.Save("models/adaBoost_gini.json")
	} else {
		// Entropy
		ac1 = &ch09.AdaBoostClassifier{}
		ac1.Load("models/adaBoost_entropy.json")
		// Gini
		ac2 = &ch09.AdaBoostClassifier{}
		ac2.Load("models/adaBoost_gini.json")
	}
	// Scoring tests
	rep := ac1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = ac2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
