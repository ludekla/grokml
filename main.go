package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch09-tree"
)

var train = flag.Bool("t", false, "train model before prediction")


func main() {

	flag.Parse()

	fmt.Println("Hello Naive Bayes! Train? ", *train)

	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7}, 
		{1, 9}, {8, 10}, {6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}
	
	dt := ch09.NewForestClassifier(3, ch09.Gini, 0.1)
	dt.Fit(dpoints, labels)

	fmt.Printf("%+v\n", dt.Score(dpoints, labels))

}
