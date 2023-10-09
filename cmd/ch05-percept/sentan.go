package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch05-percept"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Perceptron! Train? ", *train)

	csv := ds.NewCSVReader("data/IMDB_Dataset.csv", "sentiment", "review")
	modelfile := "models/ch05-percept/percept.json"

	dset := ds.NewDataSet[string](csv, ds.AtoA)
	// Sets all tokens to lower case.
	tokeniser := tk.NewTokeniser(true)

	var pc *ch05.Perceptron[tk.TokenMap]

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		// Fetch a machine.
		pc = ch05.NewTextPerceptron(10, 0.7)
		// Make strings into token maps.
		tmaps := tokeniser.Transform(trainSet.DPoints())
		// Learn.
		pc.Fit(tmaps, trainSet.Labels())
		// Transform test set.
		tmaps = tokeniser.Transform(testSet.DPoints())
		// Compute accuracy on testset.
		acc := pc.Score(tmaps, testSet.Labels())
		fmt.Printf("score on testset: %.3f\n", acc)
		persist.Dump(pc, modelfile)
	} else {
		// Load trained model.
		pc = ch05.NewTextPerceptron(0.0, 0.0)
		persist.Load(pc, modelfile)
	}

	tmaps := tokeniser.Transform(dset.DPoints())
	acc := pc.Score(tmaps, dset.Labels())
	fmt.Printf("score on dataset: %.3f\n", acc)
}
