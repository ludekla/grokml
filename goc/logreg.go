package main

import (
	"fmt"
	"goc/logreg"
)

func logregMain() {
	path := "../manning/Chapter_6_Logistic_Regression/IMDB_Dataset.csv"
	ds := logreg.NewDataSet(path)
	trainSet, testSet := ds.Split(0.2)
	var lr logreg.LogReg
	if *train {
		fmt.Println("Training")
		lr = logreg.NewLogReg(10, 0.7)
		lr.Fit(trainSet)
		lr.Save("../logr.json")
	} else {
		lr = logreg.FromJSON("../logr.json")
	}
	accTrain := lr.Accuracy(trainSet)
	accTest := lr.Accuracy(testSet)
	fmt.Println(accTrain, accTest)
}
