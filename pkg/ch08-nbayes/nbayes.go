package ch08

import (
	"encoding/json"
	"log"
	"os"
	"strings"
)

type Count struct {
	Ham  float64 `json:"ham"`
	Spam float64 `json:"spam"`
}

type NaiveBayes struct {
	vocab     map[string]Count
	count     Count
	threshold float64
}

type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

func NewNaiveBayes(th float64) NaiveBayes {
	return NaiveBayes{vocab: make(map[string]Count), threshold: th}
}

func (nb *NaiveBayes) Set(th float64) {
	nb.threshold = th
}

func (nb *NaiveBayes) Fit(ds DataSet) {
	var spam, ham int
	for _, example := range ds.Data {
		for _, token := range strings.Split(example.Text, " ") {
			if token == "" {
				continue
			}
			count, ok := nb.vocab[token]
			if !ok {
				count = Count{Spam: 1.0, Ham: 1.0}
			} else if example.Spam {
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
	count, ok := nb.vocab[token]
	if !ok {
		return Count{Spam: 1.0, Ham: 1.0}
	} else {
		return count
	}
}

func (nb NaiveBayes) Predict(email string) bool {
	return nb.Prob(email) > nb.threshold
}

func (nb NaiveBayes) Prob(email string) float64 {
	txt := strings.ToLower(email)
	var ham, spam float64 = 1.0, 1.0
	totalHam, totalSpam := nb.count.Ham, nb.count.Spam
	total := totalHam + totalSpam
	for _, token := range strings.Split(txt, " ") {
		count := nb.Get(token)
		ham *= count.Ham / totalHam * total
		spam *= count.Spam / totalSpam * total
	}
	return spam / (spam + ham)
}

func (nb NaiveBayes) Score(ds DataSet) Report {
	var tn, tp, fp, fn int
	for _, example := range ds.Data {
		spam := nb.Predict(example.Text)
		if spam && example.Spam {
			tp++
		} else if !spam && !example.Spam {
			tn++
		} else if !spam && example.Spam {
			fn++
		} else {
			fp++
		}
	}
	return Report{
		Accuracy:    float64(tp+tn) / float64(tp+tn+fn+fp),
		Recall:      float64(tp) / float64(tp+fn),
		Precision:   float64(tp) / float64(tp+fp),
		Specificity: float64(tn) / float64(tn+fp),
	}
}

func (nb NaiveBayes) Save(filepath string) {
	nbBytes, err := json.MarshalIndent(
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
	err = os.WriteFile(filepath, nbBytes, 0666)
	if err != nil {
		log.Fatal(err)
	}
}

func FromJSON(filepath string) NaiveBayes {
	nbBytes, err := os.ReadFile(filepath)
	if err != nil {
		log.Fatal(err)
	}
	nb := struct {
		Vocab     map[string]Count `json:"vocab"`
		Count     Count            `json:"count"`
		Threshold float64          `json:"threshold"`
	}{}
	err = json.Unmarshal(nbBytes, &nb)
	if err != nil {
		log.Fatal(err)
	}
	return NaiveBayes{
		vocab: nb.Vocab,
		count: nb.Count,
		threshold: nb.Threshold,
	}
}

func (r Report) FScore(beta float64) float64 {
	return (1.0 + beta*beta) * r.Recall * r.Precision / (beta*beta*r.Precision + r.Recall)
}
