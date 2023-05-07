package main

import (
	"flag"
	"fmt"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello Logistic Regression!", *train)


}
