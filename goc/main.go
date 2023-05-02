package main

import (
	"flag"
	"fmt"
)

var train = flag.Bool("t", false, "train model before prediction")

func main() {
	flag.Parse()

	// fmt.Println("Linear Regression: Housing Market (Area, No of Bedrooms) -> Price")
	// linregMain()

	// fmt.Println("Logistic Regression: Movie Reviews - Sentiment Analysis")
	// logregMain()
	
	// fmt.Println("Naive Bayes: Emails - Spam Classification")
	// nbayesMain()
	fmt.Println("---------------------------------------------------")
	fmt.Println("Decision Tree Classification: Predict Uni Admission")
	treeMain()

	fmt.Println("---------------------------------------------------")
	fmt.Println("Forest Classification: Predict Uni Admission")
	forestMain()

	fmt.Println("---------------------------------------------------")
	fmt.Println("AdaBoost Classification: Predict Uni Admission")
	adaMain()
}
