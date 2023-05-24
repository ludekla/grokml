package utils

import (
	"math"
)

type DataStats struct {
	XMean Vector  `json:"xmean"`
	XStd  Vector  `json:"xstd"`
	YMean float64 `json:"ymean"`
	YStd  float64 `json:"ystd"`
}

func NewDataStats(ds DataSet[Vector]) DataStats {
	var ymean float64
	for _, y := range ds.labels {
		ymean += y
	}
	size := float64(len(ds.labels))
	ymean /= size
	var ystd float64
	for _, y := range ds.labels {
		ystd += (y - ymean) * (y - ymean)
	}
	ystd /= size
	ystd = math.Sqrt(ystd)
	xmean, xstd := VectorStats(ds.dpoints)
	return DataStats{XMean: xmean, XStd: xstd, YMean: ymean, YStd: ystd}
}

func (stats DataStats) Normalise(ds DataSet[Vector]) DataSet[Vector] {
	xmean := stats.XMean
	nf := NewVector(len(xmean))
	for i, val := range stats.XStd {
		nf[i] = 1.0 / val
	}
	ymean, ystd := stats.YMean, stats.YStd

	nds := DataSet[Vector]{
		size:    ds.size,
		dpoints: make([]Vector, ds.size),
		labels:  make([]float64, ds.size),
	}
	for i, vec := range ds.dpoints {
		nds.dpoints[i] = vec.Add(xmean.ScaMul(-1.0)).Mul(nf)
		nds.labels[i] = (ds.labels[i] - ymean) / ystd
	}
	return nds
}
