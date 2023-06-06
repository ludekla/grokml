package ch06

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

type LogReg[D DataPoint] struct {
	Updater Updater[D] `json:"updater"`
	Bias    float64    `json:"bias"`
	NEpochs int        `json:"nepochs"`
	LRate   float64    `json:"lrate"`
}

func NewTextLogReg(nEpo int, lrate float64) *LogReg[tk.TokenMap] {
	return &LogReg[tk.TokenMap]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(TokenMapUpdater),
	}
}

func NewNumLogReg(nEpo int, lrate float64) *LogReg[vc.Vector] {
	return &LogReg[vc.Vector]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(VectorUpdater),
	}
}

func (lr *LogReg[D]) Save(filepath string) error {
	lrBytes, err := json.MarshalIndent(*lr, "", "   ")
	if err != nil {
		return fmt.Errorf("cannot read model file %v", err)
	}
	err = os.WriteFile(filepath, lrBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write model file %v", err)
	}
	return nil
}

func (lr *LogReg[D]) Load(jsonfile string) error {
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

func (lr *LogReg[D]) Fit(dpoints []D, labels []float64) []float64 {
	size := len(dpoints)
	if size == 0 {
		return nil
	}
	// Initialise weights and bias.
	lr.Updater.Init(len(dpoints[0]))
	var bias float64
	errs := make([]float64, lr.NEpochs)
	nDPs := float64(size)
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for j, dpoint := range dpoints {
			pred := sigmoid(lr.Updater.Dot(dpoint) + bias)
			diff := pred - labels[j]
			lr.Updater.Update(dpoint, -lr.LRate*diff)
			bias -= lr.LRate * diff
			sum += xentropy(pred, labels[j])
		}
		errs[i] = sum / nDPs
	}
	lr.Bias = bias
	return errs
}

func (lr LogReg[D]) Predict(dpoints []D) []float64 {
	res := make([]float64, len(dpoints))
	for i, dpoint := range dpoints {
		res[i] = sigmoid(lr.Updater.Dot(dpoint) + lr.Bias)
	}
	return res
}

// Computes the accuracy.
func (lr LogReg[D]) Score(dpoints []D, labels []float64) float64 {
	var acc float64
	preds := lr.Predict(dpoints)
	for i, pred := range preds {
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(len(dpoints))
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
