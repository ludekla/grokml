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

	var fc1, fc2 *ch09.ForestClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		fc1 = ch09.NewForestClassifier(3, ch09.NewEntropy(0.5), 0.1)
		fc1.Fit(trainSet)
		fc1.Save("models/forest_entropy.json")
		// Gini
		fc2 = ch09.NewForestClassifier(3, ch09.NewGini(0.5), 0.1)
		fc2.Fit(trainSet)
		fc2.Save("models/forest_gini.json")
	} else {
		// Entropy
		fc1 = &ch09.ForestClassifier{}
		fc1.Load("models/forest_entropy.json")
		// Gini
		fc2 = &ch09.ForestClassifier{}
		fc2.Load("models/forest_gini.json")
	}
	// Scoring tests
	rep := fc1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = fc2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
