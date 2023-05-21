package ch06

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"os"
)

type LogReg struct {
	Weights TokenMap `json:"weights"`
	Bias    float64  `json:"bias"`
	NEpochs int      `json:"nepochs"`
	LRate   float64  `json:"lrate"`
}

func NewLogReg(nEpo int, lrate float64) LogReg {
	return LogReg{Bias: rand.Float64(), NEpochs: nEpo, LRate: lrate}
}

func (lr *LogReg) Save(filepath string) {
	lrBytes, err := json.MarshalIndent(*lr, "", "   ")
	if err != nil {
		log.Fatal(err)
	}
	err = os.WriteFile(filepath, lrBytes, 0666)
	if err != nil {
		log.Fatal(err)
	}
}

func FromJSON(filepath string) LogReg {
	fileBytes, err := os.ReadFile(filepath)
	if err != nil {
		log.Fatal(err)
	}
	lr := LogReg{}
	err = json.Unmarshal(fileBytes, &lr)
	if err != nil {
		log.Fatal(err)
	}
	return lr
}

func (lr *LogReg) Fit(ds DataSet) []float64 {
	weights := TokenMap{}
	for _, example := range ds.Examples {
		weights.Update(example.tokens)
	}
	for w, _ := range weights {
		weights[w] = rand.Float64()
	}
	lr.Weights = weights
	errs := make([]float64, lr.NEpochs)
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for _, example := range ds.Examples {
			err := lr.update(example.tokens, example.label)
			sum += err
		}
		errs[i] = sum / float64(ds.Size)
	}
	return errs
}

func (lr LogReg) predict(tokens TokenMap) float64 {
	var sum float64
	for w, val := range tokens {
		sum += lr.Weights[w] * val
	}
	return sigmoid(sum + lr.Bias)
}

func (lr *LogReg) update(tokens TokenMap, label string) float64 {
	pred := lr.predict(tokens)
	diff := delta(pred, label)
	for w, val := range tokens {
		lr.Weights[w] += lr.LRate * diff * val
	}
	lr.Bias += lr.LRate * diff
	return xentropy(pred, label)
}

func (lr LogReg) Accuracy(ds DataSet) float64 {
	var acc float64
	for _, example := range ds.Examples {
		pred := lr.predict(example.tokens)
		if pred > 0.5 && example.label == "positive" {
			acc++
		} else if pred < 0.5 && example.label == "negative" {
			acc++
		}
	}
	return acc / float64(ds.Size)
}

func sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func xentropy(pred float64, label string) float64 {
	if label == "positive" {
		return -math.Log(pred)
	} else {
		return -math.Log(1.0 - pred)
	}
}

func delta(pred float64, label string) float64 {
	if label == "positive" {
		return 1.0 - pred
	} else {
		return -pred
	}
}
