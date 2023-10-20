package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch03-linreg"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	pl "grokml/pkg/pipeline"
	vc "grokml/pkg/vector"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Regularised Linear Regression Pipeline!")

	pline := pl.NewPipeline[float64, vc.Vector](
		vc.NewVectoriser(true),
		ch03.NewRegLin(1e-2, 100, 0.001, 0.0),
	)

	modelfile := "models/ch03-linreg/reglinpl.json"

	csv := ds.NewCSVReader("data/Hyderabad.csv", "Price", "Area", "No. of Bedrooms")
	dset := ds.NewDataSet[float64](csv, ds.AtoF)

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		pline.Fit(trainSet.DPoints(), trainSet.Labels())
		fmt.Printf("score on training set: %.3f\n", pline.Score(trainSet.DPoints(), testSet.Labels()))
		fmt.Printf("score on test set: %.3f\n", pline.Score(testSet.DPoints(), testSet.Labels()))
		persist.Dump(pline, modelfile)
	} else {
		fmt.Println("Use already trained model")
		persist.Load(pline, modelfile)
	}

	dpoints := [][]float64{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	preds := pline.Predict(dpoints)
	for i, dpoint := range dpoints {
		fmt.Printf("Predicted: %v -> %.3f\n", dpoint, preds[i])
	}

	fmt.Printf("score on dataset: %.3f\n", pline.Score(dset.DPoints(), dset.Labels()))

}
