package ch08

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	pl "grokml/pkg/pipeline"
	tk "grokml/pkg/tokens"
)

type Count struct {
	Ham  float64 `json:"ham"`
	Spam float64 `json:"spam"`
}

type NaiveBayes struct {
	Vocab     map[string]Count `json:"vocab"`
	Count     Count            `json:"count"`
	Threshold float64          `json:"threshold"`
	report    pl.Report        `json:"-"`
}

func NewNaiveBayes(th float64) *NaiveBayes {
	return &NaiveBayes{Vocab: make(map[string]Count), Threshold: th}
}

func (nb *NaiveBayes) Fit(tmaps []tk.TokenMap, labels []float64) []float64 {
	var spam, ham int
	for i, tmap := range tmaps {
		for token, _ := range tmap {
			count, ok := nb.Vocab[token]
			if !ok {
				count = Count{Spam: 1.0, Ham: 1.0}
			} else if labels[i] == 1.0 {
				count.Spam++
				spam++
			} else {
				count.Ham++
				ham++
			}
			nb.Vocab[token] = count
		}
	}
	nb.Count = Count{Ham: float64(ham), Spam: float64(spam)}
	return nil
}

func (nb NaiveBayes) Get(token string) Count {
	if count, ok := nb.Vocab[token]; !ok {
		return Count{Spam: 1.0, Ham: 1.0}
	} else {
		return count
	}
}

func (nb NaiveBayes) Predict(tmaps []tk.TokenMap) []float64 {
	res := make([]float64, len(tmaps))
	for i, tmap := range tmaps {
		if nb.Prob(tmap) > nb.Threshold {
			res[i] = 1.0
		}
	}
	return res
}

func (nb NaiveBayes) Prob(tmap tk.TokenMap) float64 {
	var ham, spam float64 = 1.0, 1.0
	totalHam, totalSpam := nb.Count.Ham, nb.Count.Spam
	total := totalHam + totalSpam
	for token, _ := range tmap {
		count := nb.Get(token)
		ham *= count.Ham / totalHam * total
		spam *= count.Spam / totalSpam * total
	}
	return spam / (spam + ham)
}

func (nb *NaiveBayes) Score(tmaps []tk.TokenMap, labels []float64) float64 {
	var tn, tp, fp, fn int
	preds := nb.Predict(tmaps)
	for i, pred := range preds {
		if pred == 1.0 && labels[i] == 1.0 {
			tp++
		} else if pred == 0.0 && labels[i] == 0.0 {
			tn++
		} else if pred == 0.0 && labels[i] == 1.0 {
			fn++
		} else {
			fp++
		}
	}
	acc := float64(tp+tn) / float64(tp+tn+fn+fp)
	nb.report = pl.Report{
		Accuracy:    acc,
		Recall:      float64(tp) / float64(tp+fn),
		Precision:   float64(tp) / float64(tp+fp),
		Specificity: float64(tn) / float64(tn+fp),
	}
	return acc
}

func (nb NaiveBayes) Save(filepath string) {
	nBytes, err := json.MarshalIndent(nb, "", "    ")
	if err != nil {
		log.Fatal(err)
	}
	err = os.WriteFile(filepath, nBytes, 0666)
	if err != nil {
		log.Fatal(err)
	}
}

func (nb *NaiveBayes) Load(filepath string) error {
	nbBytes, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("cannot open model file: %v", err)
	}
	err = json.Unmarshal(nbBytes, nb)
	if err != nil {
		return fmt.Errorf("cannot unmarshal the model file: %v", err)
	}
	return nil
}

func (nb NaiveBayes) GetReport() pl.Report {
	return nb.report
}

func (nb *NaiveBayes) Reset() {
	nb = &NaiveBayes{Vocab: make(map[string]Count), Threshold: nb.Threshold}
}
