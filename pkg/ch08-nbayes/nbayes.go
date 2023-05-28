package ch08

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	tk "grokml/pkg/tokens"
)

type Count struct {
	Ham  float64 `json:"ham"`
	Spam float64 `json:"spam"`
}

type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

type NaiveBayes struct {
	vocab     map[string]Count
	count     Count
	threshold float64
	report    Report
}

func NewNaiveBayes(th float64) *NaiveBayes {
	return &NaiveBayes{vocab: make(map[string]Count), threshold: th}
}

func (nb NaiveBayes) GetThreshold() float64 {
	return nb.threshold
}

func (nb *NaiveBayes) SetThreshold(th float64) {
	nb.threshold = th
}

func (nb *NaiveBayes) Fit(tmaps []tk.TokenMap, labels []float64) {
	var spam, ham int
	for i, tmap := range tmaps {
		for token, _ := range tmap {
			count, ok := nb.vocab[token]
			if !ok {
				count = Count{Spam: 1.0, Ham: 1.0}
			} else if labels[i] == 1.0 {
				count.Spam++
				spam++
			} else {
				count.Ham++
				ham++
			}
			nb.vocab[token] = count
		}
	}
	nb.count = Count{Ham: float64(ham), Spam: float64(spam)}
}

func (nb NaiveBayes) Get(token string) Count {
	if count, ok := nb.vocab[token]; !ok {
		return Count{Spam: 1.0, Ham: 1.0}
	} else {
		return count
	}
}

func (nb NaiveBayes) Predict(tmaps []tk.TokenMap) []bool {
	res := make([]bool, len(tmaps))
	for i, tmap := range tmaps {
		res[i] = nb.Prob(tmap) > nb.threshold
	}
	return res
}

func (nb NaiveBayes) Prob(tmap tk.TokenMap) float64 {
	var ham, spam float64 = 1.0, 1.0
	totalHam, totalSpam := nb.count.Ham, nb.count.Spam
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
	for i, spam := range preds {
		if spam && labels[i] == 1.0 {
			tp++
		} else if !spam && labels[i] == 0.0 {
			tn++
		} else if !spam && labels[i] == 1.0 {
			fn++
		} else {
			fp++
		}
	}
	acc := float64(tp+tn) / float64(tp+tn+fn+fp)
	nb.report = Report{
		Accuracy:    acc,
		Recall:      float64(tp) / float64(tp+fn),
		Precision:   float64(tp) / float64(tp+fp),
		Specificity: float64(tn) / float64(tn+fp),
	}
	return acc
}

func (nb NaiveBayes) Save(filepath string) {
	nBytes, err := json.MarshalIndent(
		struct {
			Vocab     map[string]Count `json:"vocab"`
			Count     Count            `json:"count"`
			Threshold float64          `json:"threshold"`
		}{
			nb.vocab,
			nb.count,
			nb.threshold,
		},
		"",
		"    ",
	)
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
	nbay := struct {
		Vocab     map[string]Count `json:"vocab"`
		Count     Count            `json:"count"`
		Threshold float64          `json:"threshold"`
	}{}
	err = json.Unmarshal(nbBytes, &nbay)
	if err != nil {
		return fmt.Errorf("cannot unmarshal the model file: %v", err)
	}
	nb.vocab = nbay.Vocab
	nb.count = nbay.Count
	nb.threshold = nbay.Threshold
	return nil
}

func (nb NaiveBayes) GetReport() Report {
	return nb.report
}

func (r Report) FScore(beta float64) float64 {
	return (1.0 + beta*beta) * r.Recall * r.Precision / (beta*beta*r.Precision + r.Recall)
}

func (nb *NaiveBayes) Reset() {
	nb = &NaiveBayes{vocab: make(map[string]Count), threshold: nb.threshold}
}
