package ch09

import (
	"math"
)

// Helper Functions
func prob(examples []Example, val float64) float64 {
	var sum float64
	for _, example := range examples {
		if example.target > val {
			sum += 1
		}
	}
	return sum / float64(len(examples))
}

type Impurity interface {
	Eval(examples []Example) float64
	Value() float64
}

func computeGain(im Impurity, examples []Example, split int) float64 {
	oldVal := im.Eval(examples)
	newVal := 0.5 * (im.Eval(examples[:split]) + im.Eval(examples[split:]))
	return oldVal - newVal
}

type Entropy struct {
	value float64
}

func NewEntropy(val float64) Entropy {
	return Entropy{val}
}

func (en Entropy) Value() float64 {
	return en.value
}
 
func (en Entropy) Eval(examples []Example) float64 {
	p := prob(examples, en.value)
	if math.Abs(p-1.0) < 1e-4 || p < 1e-4 {
		return 0.0
	}
	return -p*math.Log(p) - (1.0-p)*math.Log(1.0-p)
}

type Gini struct {
	value float64
}

func NewGini(val float64) Gini {
	return Gini{val}
}

func (gn Gini) Value() float64 {
	return gn.value
}

func (gn Gini) Eval(examples []Example) float64 {
	p := prob(examples, gn.value)
	return 2.0 * p * (1.0 - p)
}

type MSE struct {}

func NewMSE() MSE {
	return MSE{}
}

func (m MSE) Value() float64 {
	return 0.0
}

func (m MSE) Eval(examples []Example) float64 {
	var mean float64
	size := float64(len(examples))
	for _, example := range examples {
		mean += example.target
	}
	mean /= size
	val := 0.0
	for _, example := range examples {
		val += (example.target - mean) * (example.target - mean)
	}
	val /= size
	return val
}