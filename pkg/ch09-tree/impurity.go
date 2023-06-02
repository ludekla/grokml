package ch09

import (
	"math"
)

const threshold = 0.5

// Helper Functions
func prob(examples []Example) float64 {
	var sum float64
	for _, example := range examples {
		if example.label > threshold {
			sum++
		}
	}
	return sum / float64(len(examples))
}

type Impurity func(examples []Example) float64

func computeGain(eval Impurity, examples []Example, split int) float64 {
	oldVal := eval(examples)
	newVal := 0.5 * (eval(examples[:split]) + eval(examples[split:]))
	return oldVal - newVal
}
 
func Entropy(examples []Example) float64 {
	p := prob(examples)
	if math.Abs(p-1.0) < 1e-4 || p < 1e-4 {
		return 0.0
	}
	return -p*math.Log(p) - (1.0-p)*math.Log(1.0-p)
}

func Gini(examples []Example) float64 {
	p := prob(examples)
	return 2.0 * p * (1.0 - p)
}

func MSE(examples []Example) float64 {
	var mean float64
	size := float64(len(examples))
	for _, example := range examples {
		mean += example.label
	}
	mean /= size
	var val float64
	for _, example := range examples {
		val += (example.label - mean) * (example.label - mean)
	}
	val /= size
	return val
}