package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch06-logreg"
	"grokml/pkg/ch09-tree"
	"grokml/pkg/ch12-ensemble"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	pl "grokml/pkg/pipeline"
	vc "grokml/pkg/vector"
)

var train = flag.Bool("t", false, "train models before prediction")

func main() {

	flag.Parse()

	fmt.Println("Hello! Train? ", *train)

	csv := ds.NewCSVReader("data/titanic.csv", "Survived")
	dset := ds.NewDataSet[float64](csv, ds.AtoF)
	trainSet, testSet := dset.Split(0.2)

	trainPts, trainLbs := trainSet.DPoints(), trainSet.Labels()
	testPts, testLbs := testSet.DPoints(), testSet.Labels()

	var lc *ch06.LogReg[vc.Vector]
	sc := vc.NewScaler()
	veciser := vc.NewVectoriser(true)
	logreg := "models/ch06-logreg/logr_titanic.json"
	scaler := "models/ch06-logreg/scaler_titanic.json"

	var dt ch09.TreeClassifier
	tree := "models/ch09-tree/titanic_tree.json"

	var fc *ch09.ForestClassifier
	forest := "models/ch09-tree/titanic_forest.json"

	var ac *ch12.AdaBoostClassifier
	booster := "models/ch09-tree/titanic_booster.json"

	if *train {
		// Logistic Regression
		lc = ch06.NewNumLogReg(10, 0.1)
		dps := veciser.Transform(trainPts)
		sc.Fit(dps)
		dps = sc.Transform(dps)
		lc.Fit(dps, trainLbs)
		// persist.Dump(lc, logreg)
		// persist.Dump(sc, scaler)
		// decision tree
		dt = ch09.NewTreeClassifier(ch09.Entropy, 0.1)
		dt.Fit(trainPts, trainLbs)
		// persist.Dump(&dt, tree)
		// forest classifier
		fc = ch09.NewForestClassifier(3, ch09.Entropy, 0.1)
		fc.Fit(trainPts, trainLbs)
		// persist.Dump(fc, forest)
		// AdaBoost classifier
		ac = ch12.NewAdaBoostClassifier(3, ch09.Entropy, 0.1)
		ac.Fit(trainPts, trainLbs)
		// persist.Dump(ac, booster)
	} else {
		// Logistic Regression
		lc = ch06.NewNumLogReg(0.0, 0.0)
		persist.Load(lc, logreg)
		persist.Load(sc, scaler)
		// decision tree
		dt = ch09.TreeClassifier{}
		persist.Load(&dt, tree)
		// forest classifier
		fc = &ch09.ForestClassifier{}
		persist.Load(fc, forest)
		// AdaBoost classifier
		ac = &ch12.AdaBoostClassifier{}
		persist.Load(ac, booster)
	}

	// Logistic Regression classifier
	dps := veciser.Transform(testPts)
	dps = sc.Transform(dps)
	lc.Score(dps, testLbs)
	preds := lc.Predict(dps)
	rep := pl.GetReport(preds, testLbs)
	fmt.Println("Logistic Regression Classifier")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
	// decision tree
	dt.Score(testPts, testLbs)
	rep = dt.Report
	fmt.Println("Decision Tree - Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
	// forest classifier
	fc.Score(testPts, testLbs)
	rep = fc.Report
	fmt.Println("Random Forest Classifier - Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))
	// AdaBoost classifier
	ac.Score(testPts, testLbs)
	rep = ac.Report
	fmt.Println("AdaBoost Forest Classifier - Impurity: Entropy")
	fmt.Printf("report: %+v F-Score: %.4f\n", rep, rep.FScore(1.0))

}
