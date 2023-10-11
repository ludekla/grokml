package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"

	"grokml/pkg/ch08-nbayes"
	ds "grokml/pkg/dataset"
	"grokml/pkg/persist"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func ROC(model ch08.NaiveBayes, dset ds.DataSet[string], N int) {
	file, err := os.Create("data/roc.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	tokeniser := tk.NewTokeniser(true)
	tmaps := tokeniser.Transform(dset.DPoints())

	writer := csv.NewWriter(file)
	for i := 0; i <= N; i++ {
		th := float64(i) / float64(N)
		model.Threshold = th
		model.Score(tmaps, dset.Labels())
		rep := model.GetReport()
		sensi := fmt.Sprintf("%v", rep.Recall)
		speci := fmt.Sprintf("%v", rep.Specificity)
		writer.Write([]string{sensi, speci})
	}
	writer.Flush()
}

func main() {
	flag.Parse()

	csv := ds.NewCSVReader("data/emails.csv", "spam", "text")
	dset := ds.NewDataSet[string](csv, ds.AtoA)

	tokeniser := tk.NewTokeniser(true)

	var nb *ch08.NaiveBayes
	modelfile := "models/ch08-nbayes/nb.json"

	if *train {
		trainSet, testSet := dset.Split(0.2)
		fmt.Printf("Training model on %d examples.\n", trainSet.Size())

		tmaps := tokeniser.Transform(trainSet.DPoints())
		nb = ch08.NewNaiveBayes(0.5)
		nb.Fit(tmaps, trainSet.Labels())
		tmaps = tokeniser.Transform(testSet.DPoints())
		nb.Score(tmaps, testSet.Labels())
		rep := nb.GetReport()
		fmt.Printf("testset: %d emails\n", testSet.Size())
		fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))
		persist.Dump(nb, modelfile)
	} else {
		nb = &ch08.NaiveBayes{}
		persist.Load(nb, modelfile)
	}

	tmaps := tokeniser.Transform(dset.DPoints())
	nb.Score(tmaps, dset.Labels())
	rep := nb.GetReport()
	fmt.Printf("dataset: %d emails\n", dset.Size())
	fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))

	mails := [][]string{
		{"Hi Mom how are you"},
		{"meet me at the lobby of the hotel at nine am"},
		{"buy cheap lottery easy money now"},
		{"asdfgh"},
	}
	tmaps = tokeniser.Transform(mails)

	verbal := func(label float64) string {
		if label == 1.0 {
			return "Spam"
		} else {
			return "Ham"
		}
	}
	preds := nb.Predict(tmaps)
	for i, tmap := range tmaps {
		fmt.Printf("%v -> %.3f (%s)\n", mails[i], nb.Prob(tmap), verbal(preds[i]))
	}
	// ROC(nb, testSet, 10000)
}
