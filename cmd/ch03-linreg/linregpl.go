package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch03-linreg"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	"grokml/pkg/pipeline"
	vc "grokml/pkg/vector"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Linear Regression Pipeline!")

	// Linear Regression: Learning rate, number of epochs
	pl := pipeline.NewPipeline[float64, vc.Vector](
		vc.NewVectoriser(true), // transformer
		vc.NewScaler(),
		ch03.NewLinReg(1e-2, 1000), // vectoriser
	)

	modelfile := "models/ch03-linreg/linregpl.json"

	csv := ds.NewCSVReader("data/Hyderabad.csv", "Price", "Area", "No. of Bedrooms")
	dset := ds.NewDataSet[float64](csv, ds.AtoF)

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		pl.Fit(trainSet.DPoints(), trainSet.Labels())
		fmt.Printf("score on training set: %.3f\n", pl.Score(trainSet.DPoints(), trainSet.Labels()))
		fmt.Printf("score on test set: %.3f\n", pl.Score(testSet.DPoints(), testSet.Labels()))
		persist.Dump(pl, modelfile)
	} else {
		fmt.Println("Use already trained model")
		persist.Load(pl, modelfile)
	}

	dpoints := [][]float64{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	preds := pl.Predict(dpoints)
	for i, dpoint := range dpoints {
		fmt.Printf("Predicted: %v -> %.3f\n", dpoint, preds[i])
	}

	fmt.Printf("score on dataset: %.3f\n", pl.Score(dset.DPoints(), dset.Labels()))

}
