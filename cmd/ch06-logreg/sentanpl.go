package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch06-logreg"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	pl "grokml/pkg/pipeline"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Logistic Regression! Train? ", *train)

	csv := ds.NewCSVReader("data/IMDB_Dataset.csv", "sentiment", "review")
	dset := ds.NewDataSet[string](csv, ds.AtoA)

	pline := pl.NewPipeline[string, tk.TokenMap](
		tk.NewTokeniser(true),
		tk.NewNonScaler(),
		ch06.NewTextLogReg(10, 0.7),
	)

	modelfile := "models/ch06-logreg/logrpl.json"

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)

		pline.Fit(trainSet.DPoints(), trainSet.Labels())

		fmt.Printf("score on training set: %.3f\n", pline.Score(trainSet.DPoints(), trainSet.Labels()))
		fmt.Printf("score on testset: %.3f\n", pline.Score(testSet.DPoints(), testSet.Labels()))
		persist.Dump(pline, modelfile)
	} else {
		// Load trained model.
		fmt.Println("Using trained model.")
		persist.Load(pline, modelfile)
	}

	acc := pline.Score(dset.DPoints(), dset.Labels())
	fmt.Printf("score on dataset (size: %d): %.3f\n", dset.Size(), acc)
}
