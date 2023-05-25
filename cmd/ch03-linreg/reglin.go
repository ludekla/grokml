package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch03-linreg"
	"grokml/pkg/utils"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Regularised Linear Regression!")

	var lr ch03.Regression

	modelfile := "models/reglin.json"

	if *train {
		fmt.Println("Training")
		csv := utils.NewCSVReader("data/Hyderabad.csv", "Price", "Area", "No. of Bedrooms")
		dset := utils.NewDataSet[float64](csv, utils.AtoF)
		trainSet, testSet := dset.Split(0.1)
		// Learning rate, number of epochs
		lr = ch03.NewRegLin(1e-2, 100, 0.001, 0.0)
		lr.Fit(trainSet.DPoints(), trainSet.Labels())
		fmt.Printf("test score: %.3f\n", lr.Score(testSet.DPoints(), testSet.Labels()))
		lr.Save(modelfile)
	} else {
		fmt.Println("Use already trained model")
		lr = ch03.RegLinFromJSON(modelfile)
	}

	dps := [][]float64{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	preds := lr.Predict(dps)
	for i, dp := range dps {
		fmt.Printf("Predicted: %v -> %.3f\n", dp, preds[i])
	}

}
