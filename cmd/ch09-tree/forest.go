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
		"Chance of Admit",            // columns to pick
		"GRE Score", "TOEFL Score", "University Rating",
		"SOP", "LOR", "CGPA", "Research",
	)
	dset := ds.NewDataSet(csv, ds.AtoF)
	trainSet, testSet := dset.Split(0.1)

	var fc1, fc2 *ch09.ForestClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", dset.Header(), dset.Size())
		// Entropy
		fc1 = ch09.NewForestClassifier(3, ch09.Entropy, 0.1)
		fc1.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(fc1, "models/ch09-tree/forest_entropy.json")
		// Gini
		fc2 = ch09.NewForestClassifier(3, ch09.Gini, 0.1)
		fc2.Fit(trainSet.DPoints(), trainSet.Labels())
		persist.Dump(fc2, "models/ch09-tree/forest_gini.json")
	} else {
		// Entropy
		fc1 = &ch09.ForestClassifier{}
		persist.Load(fc1, "models/ch09-tree/forest_entropy.json")
		// Gini
		fc2 = &ch09.ForestClassifier{}
		persist.Load(fc2, "models/ch09-tree/forest_gini.json")
	}
	// Scoring tests
	fc1.Score(testSet.DPoints(), testSet.Labels())
	rep := fc1.Report
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	fc2.Score(testSet.DPoints(), testSet.Labels())
	rep = fc2.Report
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}
