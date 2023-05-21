package main

import (
	"flag"
	"fmt"
	
	"grokml/pkg/ch06-logreg"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Logistic Regression! Train? ", *train)

	path := "data/IMDB_Dataset.csv"
	modelfile := "models/logr.json"

	ds := ch06.NewDataSet(path)
	trainSet, testSet := ds.Split(0.1)
	var lr ch06.LogReg

	if *train {
		fmt.Println("Training")
		lr = ch06.NewLogReg(10, 0.7)
		lr.Fit(trainSet)
		lr.Save(modelfile)
	} else {
		lr = ch06.FromJSON(modelfile)
	}

	accTrain := lr.Accuracy(trainSet)
	accTest := lr.Accuracy(testSet)
	fmt.Println(accTrain, accTest)
}
