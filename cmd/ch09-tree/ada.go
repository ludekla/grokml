package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch09-tree"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	csv := ds.NewCSVReader(
		"data/Admission_Predict.csv", // data
		"Chance of Admit",            // columns
		"GRE Score", "TOEFL Score", "University Rating",
		"SOP", "LOR", "CGPA", "Research",
	)
	dset := ds.NewDataSet(csv, ds.AtoF)
	trainSet, testSet := dset.Split(0.1)

	var ac1, ac2 *ch09.AdaBoostClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", dset.Header(), dset.Size())
		// Entropy
		ac1 = ch09.NewAdaBoostClassifier(3, ch09.Entropy, 0.1)
		ac1.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(ac1, "models/ch09-tree/adaBoost_entropy.json")
		// Gini
		ac2 = ch09.NewAdaBoostClassifier(3, ch09.Gini, 0.1)
		ac2.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(ac2, "models/ch09-tree/adaBoost_gini.json")
	} else {
		// Entropy
		ac1 = &ch09.AdaBoostClassifier{}
		persist.Load(ac1, "models/ch09-tree/adaBoost_entropy.json")
		// Gini
		ac2 = &ch09.AdaBoostClassifier{}
		persist.Load(ac2, "models/ch09-tree/adaBoost_gini.json")
	}
	// Scoring tests
	ac1.Score(testSet.DPoints(), testSet.Labels())
	rep := ac1.Report
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	ac2.Score(testSet.DPoints(), testSet.Labels())
	rep = ac2.Report
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
