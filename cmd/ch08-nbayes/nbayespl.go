package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"

	"grokml/pkg/ch08-nbayes"
	ds "grokml/pkg/dataset"
	pl "grokml/pkg/pipeline"
	tk "grokml/pkg/tokens"
)

var train = flag.Bool("t", false, "train model before prediction")

func ROC(model ch08.NaiveBayes, dset ds.DataSet[string], N int) {
	file, err := os.Create("data/nbplroc.csv")
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

	pline := pl.NewPipeline[string, tk.TokenMap](
		tk.NewTokeniser(true),
		ch08.NewNaiveBayes(0.5),
	)

	modelfile := "models/ch08-nbayes/nbpl.json"

	if *train {
		trainSet, testSet := dset.Split(0.2)
		fmt.Printf("Training model on %d examples.\n", trainSet.Size())

		pline.Fit(trainSet.DPoints(), trainSet.Labels())

		var nb *ch08.NaiveBayes = pline.Estimator.(*ch08.NaiveBayes)

		pline.Score(trainSet.DPoints(), trainSet.Labels())
		rep := nb.GetReport()
		fmt.Printf("training set: %d emails\n", trainSet.Size())
		fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))

		pline.Score(testSet.DPoints(), testSet.Labels())
		rep = nb.GetReport()
		fmt.Printf("test set: %d emails\n", testSet.Size())
		fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))

		pline.Save(modelfile)
	} else {
		fmt.Println("Using trained model.")
		pline.Load(modelfile)
	}

	var nb *ch08.NaiveBayes = pline.Estimator.(*ch08.NaiveBayes)

	pline.Score(dset.DPoints(), dset.Labels())
	rep := nb.GetReport()
	fmt.Printf("dataset: %d emails\n", dset.Size())
	fmt.Printf("%+v\nF-Score: %.3f\n", rep, rep.FScore(1.0))

	mails := [][]string{
		{"Hi Mom how are you"},
		{"meet me at the lobby of the hotel at nine am"},
		{"buy cheap lottery easy money now"},
		{"asdfgh"},
	}

	verbal := func(pred float64) string {
		if pred == 1.0 {
			return "Spam"
		} else {
			return "Ham"
		}
	}
	preds := pline.Predict(mails)

	tmaps := pline.Transformer.Transform(mails)

	for i, tmap := range tmaps {
		fmt.Printf("%v -> %.3f (%s)\n", mails[i], nb.Prob(tmap), verbal(preds[i]))
	}
	// ROC(nb, testSet, 10000)

}
