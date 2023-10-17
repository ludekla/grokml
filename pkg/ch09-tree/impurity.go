package ch09

import (
	"math"
)

const threshold = 0.5

// prob is a helper function to compute the relative frequency of examples
// whose label surpasses the threshold.
func prob(examples []Example) float64 {
	var sum float64
	for _, example := range examples {
		if example.label > threshold {
			sum++
		}
	}
	return sum / float64(len(examples))
}

// Mean is a helper function to compute the average value of a slice.
func Mean(vals []float64) float64 {
	var mean float64
	for _, val := range vals {
		mean += val
	}
	mean /= float64(len(vals))
	return mean
}

// Impurity is a function type that computes the impurity
// of the given set of examples.
type Impurity func(examples []Example) float64

// computeGain calculates the loss of impurity of a given split where
// the split is characterised by its slice index.
func computeGain(eval Impurity, examples []Example, split int) float64 {
	oldVal := eval(examples)
	newVal := 0.5 * (eval(examples[:split]) + eval(examples[split:]))
	return oldVal - newVal
}

// Entropy implements the impurity function type suitable for classifications.
func Entropy(examples []Example) float64 {
	p := prob(examples)
	if math.Abs(p-1.0) < 1e-4 || p < 1e-4 {
		return 0.0
	}
	return -p*math.Log(p) - (1.0-p)*math.Log(1.0-p)
}

// Gini implements the impurity function type. It is suitable for classifications.
func Gini(examples []Example) float64 {
	p := prob(examples)
	return 2.0 * p * (1.0 - p)
}

// MSE implements the impurity function type. It is suitable for regression, aka
// mean-squared error.
func MSE(examples []Example) float64 {
	var mean float64 // compute the mean
	size := float64(len(examples))
	for _, example := range examples {
		mean += example.label
	}
	mean /= size
	var val float64 // compute the standard deviation
	for _, example := range examples {
		val += (example.label - mean) * (example.label - mean)
	}
	val /= size
	return val
}
