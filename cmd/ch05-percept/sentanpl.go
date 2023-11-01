package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch05-percept"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	pl "grokml/pkg/pipeline"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Perceptron Pipeline! Train? ", *train)

	csv := ds.NewCSVReader("data/IMDB_Dataset.csv", "sentiment", "review")
	dset := ds.NewDataSet[string](csv, ds.AtoA)

	modelfile := "models/ch05-percept/perceptpl.json"

	pline := pl.NewPipeline[string, tk.TokenMap](
		tk.NewTokeniser(true),
		tk.NewNonScaler(),
		ch05.NewTextPerceptron(10, 0.7),
	)

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)

		pline.Fit(trainSet.DPoints(), trainSet.Labels())

		fmt.Printf("score on training set: %.3f\n", pline.Score(trainSet.DPoints(), trainSet.Labels()))
		fmt.Printf("score on testset: %.3f\n", pline.Score(testSet.DPoints(), testSet.Labels()))
		persist.Dump(pline, modelfile)
	} else {
		// Load trained model.
		fmt.Println("Using already trained model.")
		persist.Load(pline, modelfile)
	}

	fmt.Printf("score on dataset: %.3f\n", pline.Score(dset.DPoints(), dset.Labels()))
}
