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

	fmt.Println("Hello Regularised Linear Regression!")

	var lr *ch03.RegLin

	vectoriser := vc.NewVectoriser(true)

	modelfile := "models/ch03-linreg/reglin.json"
	scalerFile := "models/ch03-linreg/reglin_scaler.json"

	var dpoints []vc.Vector

	csv := ds.NewCSVReader("data/Hyderabad.csv", "Price", "Area", "No. of Bedrooms")
	dset := ds.NewDataSet[float64](csv, ds.AtoF)
	scaler := vc.NewScaler()

	if *train {
		fmt.Println("Training")
		trainSet, testSet := dset.Split(0.1)
		dpoints = vectoriser.Transform(trainSet.DPoints())
		scaler.Fit(dpoints)
		// Learning rate, number of epochs
		lr = ch03.NewRegLin(1e-2, 100, 0.001, 0.0)
		dpoints = scaler.Transform(dpoints)
		lr.Fit(dpoints, trainSet.Labels())
		dpoints = vectoriser.Transform(testSet.DPoints())
		dpoints = scaler.Transform(dpoints)
		fmt.Printf("score on testset: %.3f\n", lr.Score(dpoints, testSet.Labels()))
		persist.Dump(lr, modelfile)
		persist.Dump(scaler, scalerFile)
	} else {
		fmt.Println("Use already trained model")
		lr = &ch03.RegLin{}
		persist.Load(lr, modelfile)
		persist.Load(scaler, scalerFile)
	}

	vecs := []vc.Vector{{600, 1}, {1000, 2}, {1500, 3}, {2000, 4}}
	vecs = scaler.Transform(vecs)
	preds := lr.Predict(vecs)
	for i, vec := range vecs {
		fmt.Printf("Predicted: %v -> %.3f\n", vec, preds[i])
	}

	dpoints = vectoriser.Transform(dset.DPoints())
	dpoints = scaler.Transform(dpoints)
	fmt.Printf("score on dataset: %.3f\n", lr.Score(dpoints, dset.Labels()))

}
