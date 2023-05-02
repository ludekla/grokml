package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"goc/nbayes"
)

func ROC(model nbayes.NaiveBayes, ds nbayes.DataSet, N int) {
	file, err := os.Create("roc.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	for i := 0; i <= N; i++ {
		th := float64(i) / float64(N)
		model.Set(th)
		rep := model.Score(ds)
		sensi := fmt.Sprintf("%v", rep.Recall)
		speci := fmt.Sprintf("%v", rep.Specificity)
		writer.Write([]string{sensi, speci})
	}
	writer.Flush()
}

func nbayesMain() {
	ds := nbayes.NewDataSet("../manning/Chapter_8_Naive_Bayes/emails.csv")
	trainSet, testSet := ds.Split(0.2)
	var nb nbayes.NaiveBayes
	if *train {
		fmt.Printf("Training model on %d examples.\n", trainSet.Size)
		nb = nbayes.NewNaiveBayes(0.5)
		nb.Fit(trainSet)
		nb.Save("nb.json")
	} else {
		nb = nbayes.FromJSON("nb.json")
	}
	rep := nb.Score(testSet)
	fmt.Printf("testset: %d emails\n", testSet.Size)
	fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))

	mails := []string{
		"Hi Mom how are you",
		"meet me at the lobby of the hotel at nine am",
		"buy cheap lottery easy money now",
		"asdfgh",
	}
	verbal := func(v bool) string {
		if v {
			return "Spam"
		} else {
			return "Ham"
		}
	}
	for _, mail := range mails {
		fmt.Printf("%s -> %.3f (%s)\n", mail, nb.Prob(mail), verbal(nb.Predict(mail)))
	}
	// ROC(nb, testSet, 10000)
}
