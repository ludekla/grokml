package ch08

import (
	"encoding/json"

	pl "grokml/pkg/pipeline"
	tk "grokml/pkg/tokens"
)

// Count serves as a container for ham and spam counts which are assigned to
// each word encountered during processing. We use floats to ease calculations
// involving float division.
type Count struct {
	Ham  float64 `json:"ham"`
	Spam float64 `json:"spam"`
}

// NaiveBayes implements a Naive-Bayes classifier.
type NaiveBayes struct {
	Vocab     map[string]Count `json:"vocab"`
	Count     Count            `json:"count"`
	Threshold float64          `json:"threshold"`
	report    pl.Report        `json:"-"`
}

// NewNaiveBayes is a factory function for pointers to NaiveBayes objects.
func NewNaiveBayes(th float64) *NaiveBayes {
	return &NaiveBayes{Vocab: make(map[string]Count), Threshold: th}
}

// Fit performs training on email documents as token maps.
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

// Get returns the spam and ham count for a given token. The zero count
// is given by an equal count of 1.0 for both ham and spam to avoid division
// by zero.
func (nb NaiveBayes) Get(token string) Count {
	if count, ok := nb.Vocab[token]; !ok {
		return Count{Spam: 1.0, Ham: 1.0}
	} else {
		return count
	}
}

// Predict classifies the given emails as token maps.
func (nb NaiveBayes) Predict(tmaps []tk.TokenMap) []float64 {
	res := make([]float64, len(tmaps))
	for i, tmap := range tmaps {
		if nb.Prob(tmap) > nb.Threshold {
			res[i] = 1.0
		}
	}
	return res
}

// Prob computes the probability of a given email to be spam.
// The variable 'total' guarantees computational stability and is
// cancelled out in the end result.
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

// Score computes not just the accuracy and thereby satisfies the Estimator
// interface as defined in the pipeline pkg, but also sets a Report object
// with all the necessary quantities to compute the F score.
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

// Marshal and Unmarshal implement the JSONable interface
func (nb NaiveBayes) Marshal() ([]byte, error) {
	return json.MarshalIndent(nb, "", "    ")
}

func (nb *NaiveBayes) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, nb)
}

// GetReport is a getter method for the report field.
func (nb NaiveBayes) GetReport() pl.Report {
	return nb.report
}
