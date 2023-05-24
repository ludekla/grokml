package ch03

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"grokml/pkg/utils"
)

type Regression interface {
	Fit(ds utils.DataSet[utils.Vector]) []float64
	Predict(points []utils.Vector) []float64
	Score(ds utils.DataSet[utils.Vector]) float64
	Save(filepath string) error
}

type LinReg struct {
	Weights utils.Vector `json:"weights"`
	Bias    float64      `json:"bias"`
	LRate   float64      `json:"lrate"`
	NEpochs int          `json:"nepochs"`
}

func NewLinReg(lr float64, epochs int) *LinReg {
	return &LinReg{LRate: lr, NEpochs: epochs}
}

func (l *LinReg) Fit(ds utils.DataSet[utils.Vector]) []float64 {
	stats := utils.NewDataStats(ds)
	nds := stats.Normalise(ds)
	l.Weights = utils.RandVector(len(stats.XMean))
	l.Bias = rand.Float64()
	errs := make([]float64, 0, l.NEpochs)
	size := nds.Size()
	for ep := 0; ep < l.NEpochs; ep++ {
		var err float64
		for i := 0; i < size; i++ {
			x, y := nds.Random()
			delta := l.Weights.Dot(x) + l.Bias - y
			l.Weights.IAdd(x.ScaMul(-l.LRate * delta))
			l.Bias -= l.LRate * delta
			err += delta * delta
		}
		errs = append(errs, math.Sqrt(err/float64(size)))
	}
	l.Weights.IDiv(stats.XStd)
	l.Weights.IScaMul(stats.YStd)
	l.Bias = stats.YMean + stats.YStd*l.Bias - l.Weights.Dot(stats.XMean)
	return errs
}

func (l LinReg) Predict(points []utils.Vector) []float64 {
	preds := make([]float64, len(points))
	for i, point := range points {
		preds[i] = l.Weights.Dot(point) + l.Bias
	}
	return preds
}

// Computes the coefficient of determination
func (l LinReg) Score(ds utils.DataSet[utils.Vector]) float64 {
	preds := l.Predict(ds.X())
	ym := mean(preds)
	// Residual Sum of Squares, Total Sum of Squares
	var rss, tss float64
	y := ds.Y()
	for i, pred := range preds {
		rss += (y[i] - pred) * (y[i] - pred)
		tss += (y[i] - ym) * (y[i] - ym)
	}
	return 1.0 - rss/tss
}

func mean(numbers []float64) float64 {
	var m float64
	for _, val := range numbers {
		m += val
	}
	return m / float64(len(numbers))
}

func (l LinReg) Save(filepath string) error {
	asBytes, err := json.MarshalIndent(l, "", "    ")
	if err != nil {
		return fmt.Errorf("cannot marshal %v into JSON bytes", l)
	}
	err = os.WriteFile(filepath, asBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write JSON bytes into file %s", filepath)
	}
	return nil
}

func LinRegFromJSON(filepath string) *LinReg {
	lr := LinReg{}
	asBytes, err := os.ReadFile(filepath)
	if err != nil {
		log.Fatalf("cannot read from file %s: %v", filepath, err)
	}
	err = json.Unmarshal(asBytes, &lr)
	if err != nil {
		log.Fatalf("cannot unmarshal JSON bytes: %v", err)
	}
	return &lr
}
