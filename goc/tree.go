package main

import (
	"fmt"

	dec "goc/tree"
)

func treeMain() {
	ds := dec.NewDataSet("../manning/Chapter_9_Decision_Trees/Admission_Predict.csv")
	trainSet, testSet := ds.Split(0.1)
	
	var dt1, dt2 dec.TreeClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		dt1 = dec.NewTreeClassifier(dec.NewImpurity(0.5, dec.Entropy), 0.1)
		dt1.Fit(trainSet)
		dt1.Save("tree/jsons/tree_entropy.json")
		// Gini
		dt2 = dec.NewTreeClassifier(dec.NewImpurity(0.5, dec.Gini), 0.1)
		dt2.Fit(trainSet)
		dt2.Save("tree/jsons/tree_gini.json")
	} else {
		// Entropy
		dt1 = dec.TreeClassifier{}
		dt1.Load("tree/jsons/tree_entropy.json")
		// Gini
		dt2 = dec.TreeClassifier{}
		dt2.Load("tree/jsons/tree_gini.json")
	}
	// Scoring tests
	rep := dt1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = dt2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}

func forestMain() {
	ds := dec.NewDataSet("../manning/Chapter_9_Decision_Trees/Admission_Predict.csv")
	trainSet, testSet := ds.Split(0.1)
	
	var fc1, fc2 *dec.ForestClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		fc1 = dec.NewForestClassifier(3, dec.NewImpurity(0.5, dec.Entropy), 0.1)
		fc1.Fit(trainSet)
		fc1.Save("tree/jsons/forest_entropy.json")
		// Gini
		fc2 = dec.NewForestClassifier(3, dec.NewImpurity(0.5, dec.Gini), 0.1)
		fc2.Fit(trainSet)
		fc2.Save("tree/jsons/forest_gini.json")
	} else {
		// Entropy
		fc1 = &dec.ForestClassifier{}
		fc1.Load("tree/jsons/forest_entropy.json")
		// Gini
		fc2 = &dec.ForestClassifier{}
		fc2.Load("tree/jsons/forest_gini.json")
	}
	// Scoring tests
	rep := fc1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = fc2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}

func adaMain() {
	ds := dec.NewDataSet("../manning/Chapter_9_Decision_Trees/Admission_Predict.csv")
	trainSet, testSet := ds.Split(0.1)
	
	var ac1, ac2 *dec.AdaBoostClassifier
	// var dt3 TreeRegressor

	if *train {
		fmt.Printf("Training on Dataset\nheader: %v size: %d\n", ds.Header, ds.Size)
		// Entropy
		ac1 = dec.NewAdaBoostClassifier(3, dec.NewImpurity(0.5, dec.Entropy), 0.1)
		ac1.Fit(trainSet)
		ac1.Save("tree/jsons/adaBoost_entropy.json")
		// Gini
		ac2 = dec.NewAdaBoostClassifier(3, dec.NewImpurity(0.5, dec.Gini), 0.1)
		ac2.Fit(trainSet)
		ac2.Save("tree/jsons/adaBoost_gini.json")
	} else {
		// Entropy
		ac1 = &dec.AdaBoostClassifier{}
		ac1.Load("tree/jsons/adaBoost_entropy.json")
		// Gini
		ac2 = &dec.AdaBoostClassifier{}
		ac2.Load("tree/jsons/adaBoost_gini.json")
	}
	// Scoring tests
	rep := ac1.Score(testSet)
	fmt.Println("Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

	rep = ac2.Score(testSet)
	fmt.Println("Impurity: Gini")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
}