package ch06

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"os"

	"grokml/pkg/utils"
)

type LogReg struct {
	Weights utils.TokenMap `json:"weights"`
	Bias    float64        `json:"bias"`
	NEpochs int            `json:"nepochs"`
	LRate   float64        `json:"lrate"`
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

func FromJSON(jsonfile string) LogReg {
	fileBytes, err := os.ReadFile(jsonfile)
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

func (lr *LogReg) Fit(ds utils.DataSet[utils.TokenMap]) []float64 {
	// initialise weights with
	weights := utils.TokenMap{}
	for _, dpoint := range ds.X() {
		for token, _ := range dpoint {
			if _, ok := weights[token]; !ok {
				weights[token] = rand.Float64()
			}
		}
	}
	lr.Weights = weights
	errs := make([]float64, lr.NEpochs)
	dpoints := ds.X()
	labels := ds.Y()
	size := float64(ds.Size())
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for j, dpoint := range dpoints {
			err := lr.update(dpoint, labels[j])
			sum += err
		}
		errs[i] = sum / size
	}
	return errs
}

func (lr LogReg) predict(tokens utils.TokenMap) float64 {
	var sum float64
	for w, val := range tokens {
		sum += lr.Weights[w] * val
	}
	return sigmoid(sum + lr.Bias)
}

func (lr *LogReg) update(tokens utils.TokenMap, label float64) float64 {
	pred := lr.predict(tokens)
	diff := label - pred
	for w, val := range tokens {
		lr.Weights[w] += lr.LRate * diff * val
	}
	lr.Bias += lr.LRate * diff
	return xentropy(pred, label)
}

func (lr LogReg) Accuracy(ds utils.DataSet[utils.TokenMap]) float64 {
	var acc float64
	labels := ds.Y()
	for i, dpoint := range ds.X() {
		pred := lr.predict(dpoint)
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(ds.Size())
}

func sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func xentropy(pred float64, label float64) float64 {
	if label == 1.0 {
		return -math.Log(pred)
	} else {
		return -math.Log(1.0 - pred)
	}
}
