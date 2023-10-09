package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch03-linreg"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	vc "grokml/pkg/vector"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Linear Regression!")

	var lr *ch03.LinReg

	vectoriser := vc.NewVectoriser(true)

	modelfile := "models/ch03-linreg/linreg.json"
	var dpoints []vc.Vector

	csv := ds.NewCSVReader("data/Hyderabad.csv", "Price", "Area", "No. of Bedrooms")
	dset := ds.NewDataSet[float64](csv, ds.AtoF)

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		dpoints = vectoriser.Transform(trainSet.DPoints())
		// Learning rate, number of epochs
		lr = ch03.NewLinReg(1e-2, 1000)
		lr.Fit(dpoints, trainSet.Labels())
		fmt.Printf("score on training set: %.3f\n", lr.Score(dpoints, trainSet.Labels()))
		dpoints = vectoriser.Transform(testSet.DPoints())
		fmt.Printf("score on testset: %.3f\n", lr.Score(dpoints, testSet.Labels()))
		persist.Dump(lr, modelfile)
	} else {
		fmt.Println("Use already trained model")
		lr = &ch03.LinReg{}
		persist.Load(lr, modelfile)
	}

	vecs := []vc.Vector{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	preds := lr.Predict(vecs)
	for i, vec := range vecs {
		fmt.Printf("Predicted: %v -> %.3f\n", vec, preds[i])
	}

	dpoints = vectoriser.Transform(dset.DPoints())
	fmt.Printf("score on dataset: %.3f\n", lr.Score(dpoints, dset.Labels()))

}
