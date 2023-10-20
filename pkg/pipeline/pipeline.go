package pipeline

import (
	"encoding/json"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// InType is the input type for the pipeline input.
type InType interface {
	float64 | string
}

// OutType is the output type for the transformer and at the same
// time the input type for the estimator.
type OutType interface {
	vc.Vector | tk.TokenMap
}

// Transformer is the interface for a data transformation engine.
type Transformer[I InType, O OutType] interface {
	Transform([][]I) []O
}

// Estimator is the generic type of all classifier and regression
// engines.
type Estimator[O OutType] interface {
	Fit(dpoints []O, labels []float64) []float64
	Predict(dpoints []O) []float64
	Score(dpoints []O, labels []float64) float64
}

// Pipeline implements the ML pipeline concept. It consists of a
// transformer that transforms the data into a form digestable
// for the estimator.
type Pipeline[I InType, O OutType] struct {
	Transformer Transformer[I, O] `json:"transformer"`
	Estimator   Estimator[O]      `json:"estimator"`
}

// Report serves as a container for 4 standard ML performance measures.
// While accuracy is both for regressors and classifiers, precision
// recall and specificity are only used for the latter.
type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

// FScore computes the standard performance measure called F-score
// aka F1-score. Its extremes beta = 0 and beta = infinity are recall and
// precision respectively. For all in-between values, it is a
// mixture. In the case beta = 1 this mixture is a quotient of the geometric
// and the arithmetic average.
func (rp Report) FScore(beta float64) float64 {
	denominator := beta*beta*rp.Recall + rp.Precision
	if denominator == 0.0 {
		return 0.0
	}
	return (1 + beta*beta) * rp.Recall * rp.Precision / denominator
}

// NewPipeline is the factory function for Pipeline.
func NewPipeline[I InType, O OutType](trf Transformer[I, O], est Estimator[O]) *Pipeline[I, O] {
	return &Pipeline[I, O]{Transformer: trf, Estimator: est}
}

// Fit implements the training of the pipeline. It returns the epoch errors.
func (pl *Pipeline[I, O]) Fit(dpoints [][]I, labels []float64) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Fit(tdpoints, labels)
}

// Predict implements the prediction method. It returns the predicted labels.
func (pl *Pipeline[I, O]) Predict(dpoints [][]I) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Predict(tdpoints)
}

// Score computes the accuracy of the estimator on the given dataset.
func (pl *Pipeline[I, O]) Score(dpoints [][]I, labels []float64) float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Score(tdpoints, labels)
}

// Marshal and Unmarshal implement the JSONable interface (pkg/persist).
func (pl Pipeline[I, O]) Marshal() ([]byte, error) {
	return json.MarshalIndent(pl, "", "   ")
}

func (pl *Pipeline[I, O]) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, pl)
}
