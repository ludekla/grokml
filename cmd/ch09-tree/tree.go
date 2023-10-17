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
		"data/Admission_Predict.csv",   // filename
		"Chance of Admit", "GRE Score", // columns to pick
		"TOEFL Score", "University Rating",
		"SOP", "LOR", "CGPA", "Research",
	)
	dset := ds.NewDataSet(csv, ds.AtoF)
	trainSet, testSet := dset.Split(0.1)

	var dt1, dt2 ch09.TreeClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", dset.Header(), dset.Size())
		// Entropy
		dt1 = ch09.NewTreeClassifier(ch09.Entropy, 0.1)
		dt1.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(&dt1, "models/ch09-tree/tree_entropy.json")
		// Gini
		dt2 = ch09.NewTreeClassifier(ch09.Gini, 0.1)
		dt2.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(&dt2, "models/ch09-tree/tree_gini.json")
	} else {
		// Entropy
		dt1 = ch09.TreeClassifier{}
		persist.Load(&dt1, "models/ch09-tree/tree_entropy.json")
		// Gini
		dt2 = ch09.TreeClassifier{}
		persist.Load(&dt2, "models/ch09-tree/tree_gini.json")
	}
	// Scoring tests
	dt1.Score(testSet.DPoints(), testSet.Labels())
	rep := dt1.Report
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	dt2.Score(testSet.DPoints(), testSet.Labels())
	rep = dt2.Report
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
