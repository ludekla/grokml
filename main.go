package main

import (
	"flag"
	"fmt"
	"os"

	ds "grokml/pkg/dataset"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Perceptron! Train? ", *train)

	tb, err := ds.NewCSVTable("data/titanic.csv")
	if err != nil {
		fmt.Printf("Cannot read csv: %v\n", err)
		os.Exit(0)
	}
	fmt.Println(tb.Header)
	for i, rec := range tb.Records[:10] {
		fmt.Println(i, rec)
	}

}
