package ch06

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	tk "grokml/pkg/tokens"
)

type LogReg struct {
	Weights tk.TokenMap `json:"weights"`
	Bias    float64     `json:"bias"`
	NEpochs int         `json:"nepochs"`
	LRate   float64     `json:"lrate"`
}

func NewLogReg(nEpo int, lrate float64) *LogReg {
	return &LogReg{Bias: rand.Float64(), NEpochs: nEpo, LRate: lrate}
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

func (lr *LogReg) Load(jsonfile string) error {
	fileBytes, err := os.ReadFile(jsonfile)
	if err != nil {
		return fmt.Errorf("cannot read model file %v", err)
	}
	err = json.Unmarshal(fileBytes, lr)
	if err != nil {
		return fmt.Errorf("cannot load model %v", err)
	}
	return nil
}

func (lr *LogReg) Fit(tmaps []tk.TokenMap, labels []float64) []float64 {
	// Initialise weights and bias.
	weights := make(tk.TokenMap)
	var bias float64
	errs := make([]float64, lr.NEpochs)
	size := float64(len(tmaps))
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for j, tmap := range tmaps {
			pred := sigmoid(weights.Dot(tmap) + bias)
			diff := pred - labels[j]
			weights.IAdd(tmap.ScaMul(-lr.LRate * diff))
			bias -= lr.LRate * diff
			sum += xentropy(pred, labels[j])
		}
		errs[i] = sum / size
	}
	lr.Weights = weights
	lr.Bias = bias
	return errs
}

func (lr LogReg) Predict(tmap tk.TokenMap) float64 {
	return sigmoid(lr.Weights.Dot(tmap) + lr.Bias)
}

// Computes the accuracy.
func (lr LogReg) Score(tmaps []tk.TokenMap, labels []float64) float64 {
	var acc float64
	for i, tmap := range tmaps {
		pred := lr.Predict(tmap)
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(len(tmaps))
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
