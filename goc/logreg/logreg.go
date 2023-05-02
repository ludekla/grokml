package logreg

import (
	"math"
	"math/rand"
)

type LogReg struct {
	Weights TokenMap
	Bias float64
	NEpochs int
	LRate float64
}

func NewLogReg(n int, lrate float64) LogReg {
	return LogReg{Bias: rand.Float64(), NEpochs: n, LRate: lrate}
}

func (lr *LogReg) Fit(ds DataSet) []float64 {
	weights := TokenMap{}
	for _, tokens := range ds.X {
		weights.Update(tokens)
	}
	for w, _ := range weights {
		weights[w] = rand.Float64()
	}
	lr.Weights = weights
	errs := make([]float64, lr.NEpochs)
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for j, tokens := range ds.X {
			err := lr.update(tokens, ds.Y[j])
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
	for i, tokens := range ds.X {
		pred := lr.predict(tokens)
		if pred > 0.5 && ds.Y[i] == "positive" {
			acc++
		} else if pred < 0.5 && ds.Y[i] == "negative" {
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
