package ch05

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"

	"grokml/pkg/ch06-logreg"
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

type Perceptron[D ch06.DataPoint] struct {
	Updater ch06.Updater[D] `json:"updater"`
	Bias    float64         `json:"bias"`
	NEpochs int             `json:"nepochs"`
	LRate   float64         `json:"lrate"`
}

func NewTextPerceptron(nEpo int, lrate float64) *Perceptron[tk.TokenMap] {
	return &Perceptron[tk.TokenMap]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(ch06.TokenMapUpdater),
	}
}

func NewNumPerceptron(nEpo int, lrate float64) *Perceptron[vc.Vector] {
	return &Perceptron[vc.Vector]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(ch06.VectorUpdater),
	}
}

func (pc *Perceptron[D]) Save(filepath string) {
	pcBytes, err := json.MarshalIndent(*pc, "", "   ")
	if err != nil {
		log.Fatal(err)
	}
	err = os.WriteFile(filepath, pcBytes, 0666)
	if err != nil {
		log.Fatal(err)
	}
}

func (pc *Perceptron[D]) Load(jsonfile string) error {
	fileBytes, err := os.ReadFile(jsonfile)
	if err != nil {
		return fmt.Errorf("cannot read model file %v", err)
	}
	err = json.Unmarshal(fileBytes, pc)
	if err != nil {
		return fmt.Errorf("cannot load model %v", err)
	}
	return nil
}

func (pc *Perceptron[D]) Fit(dpoints []D, labels []float64) []float64 {
	size := len(dpoints)
	if size == 0 {
		return nil
	}
	// Initialise weights and bias.
	pc.Updater.Init(len(dpoints[0]))
	var bias float64
	errs := make([]float64, pc.NEpochs)
	nDPs := float64(size)
	for i := 0; i < pc.NEpochs; i++ {
		var sum float64
		for j, dpoint := range dpoints {
			pred := heaviside(pc.Updater.Dot(dpoint) + bias)
			diff := pred - labels[j]
			if diff != 0.0 {
				pc.Updater.Update(dpoint, -pc.LRate*diff)
				bias -= pc.LRate * diff
			} else {
				sum++
			}
		}
		errs[i] = sum / nDPs
	}
	pc.Bias = bias
	return errs
}

func (pc Perceptron[D]) Predict(dpoints []D) []float64 {
	res := make([]float64, len(dpoints))
	for i, dpoint := range dpoints {
		res[i] = heaviside(pc.Updater.Dot(dpoint) + pc.Bias)
	}
	return res
}

// Computes the accuracy.
func (pc Perceptron[D]) Score(dpoints []D, labels []float64) float64 {
	var acc float64
	preds := pc.Predict(dpoints)
	for i, pred := range preds {
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(len(dpoints))
}

func heaviside(val float64) float64 {
	if val < 0.0 {
		return 0.0
	} else {
		return 1.0
	}
}
