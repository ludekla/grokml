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

	fmt.Println("Hello Linear Regression!")

	var lr ch03.Regression

	modelfile := "config/linreg.json"

	if *train {
		fmt.Println("Training")
		path := "data/Hyderabad.csv"
		dset := utils.NewDataSet(path, "Price", []string{"Area", "No. of Bedrooms"})
		trainSet, testSet := dset.Split(0.1)
		// Learning rate, number of epochs
		lr = ch03.NewLinReg(1e-2, 100)
		lr.Fit(trainSet)
		fmt.Printf("test score: %.3f\n", lr.Score(testSet))
		lr.Save(modelfile)
	} else {
		fmt.Println("Use already trained model")
		lr = ch03.LinRegFromJSON(modelfile)
	}

	vecs := []utils.Vector{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	preds := lr.Predict(vecs)
	for i, vec := range vecs {
		fmt.Printf("Predicted: %v -> %.3f\n", vec, preds[i])
	}

}
