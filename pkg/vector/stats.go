package vector

import (
	"math"
)

type DataStats struct {
	XMean Vector  `json:"xmean"`
	XStd  Vector  `json:"xstd"`
	YMean float64 `json:"ymean"`
	YStd  float64 `json:"ystd"`
}

func GetDataStats(vecs []Vector, labels []float64) DataStats {
	var ymean float64
	for _, y := range labels {
		ymean += y
	}
	size := float64(len(labels))
	ymean /= size
	var ystd float64
	for _, y := range labels {
		ystd += (y - ymean) * (y - ymean)
	}
	ystd /= size
	ystd = math.Sqrt(ystd)
	xmean, xstd := VectorStats(vecs)
	return DataStats{XMean: xmean, XStd: xstd, YMean: ymean, YStd: ystd}
}

func (stats DataStats) Normalise(vecs []Vector, labels []float64) ([]Vector, []float64) {
	xmean, xstd := stats.XMean, stats.XStd
	ymean, ystd := stats.YMean, stats.YStd
	outvecs := make([]Vector, len(vecs))
	outlabels := make([]float64, len(vecs))
	for i, vec := range vecs {
		outvecs[i] = vec.Add(xmean.ScaMul(-1.0)).Div(xstd)
		outlabels[i] = (labels[i] - ymean) / ystd
	}
	return outvecs, outlabels
}
