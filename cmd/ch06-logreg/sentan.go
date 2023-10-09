package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch06-logreg"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Logistic Regression! Train? ", *train)

	csv := ds.NewCSVReader("data/IMDB_Dataset.csv", "sentiment", "review")
	modelfile := "models/ch06-logreg/logr.json"

	dset := ds.NewDataSet[string](csv, ds.AtoA)
	// Sets all tokens to lower case.
	tokeniser := tk.NewTokeniser(true)

	var lr *ch06.LogReg[tk.TokenMap]

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		// Fetch a machine.
		lr = ch06.NewTextLogReg(10, 0.7)
		// Make strings into token maps.
		tmaps := tokeniser.Transform(trainSet.DPoints())
		// Learn.
		lr.Fit(tmaps, trainSet.Labels())
		// Transform test set.
		tmaps = tokeniser.Transform(testSet.DPoints())
		// Compute accuracy on testset.
		acc := lr.Score(tmaps, testSet.Labels())
		fmt.Printf("score on testset: %.3f\n", acc)
		persist.Dump(lr, modelfile)
	} else {
		// Load trained model.
		lr = ch06.NewTextLogReg(0.0, 0.0)
		persist.Load(lr, modelfile)
	}

	tmaps := tokeniser.Transform(dset.DPoints())
	acc := lr.Score(tmaps, dset.Labels())
	fmt.Printf("score on dataset: %.3f\n", acc)
}
