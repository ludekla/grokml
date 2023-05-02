package vaux

import (
	"encoding/csv"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type DataSet struct {
	Path string
	Size int
	X    []Vector
	Y    []float64
}

type DataStats struct {
	XMean Vector  `json:"xmean"`
	XStd  Vector  `json:"xstd"`
	YMean float64 `json:"ymean"`
	YStd  float64 `json:"ystd"`
}

// fetch data from csv file with specified columns
func NewDataSet(csvfile string, target string, features []string) DataSet {
	ds := DataSet{Path: "../" + csvfile}
	file, err := os.Open(ds.Path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	cr := csv.NewReader(file)
	rec, err := cr.Read()
	if err != nil {
		log.Fatal(err)
	}
	targetCol, featureCols := findCols(rec, target, features)
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		row := NewVector(len(featureCols))
		for j, idx := range featureCols {
			x, err := strconv.ParseFloat(rec[idx], 64)
			if err != nil {
				log.Fatal(err)
			}
			row[j] = x
		}
		ds.X = append(ds.X, row)
		y, err := strconv.ParseFloat(rec[targetCol], 64)
		if err != nil {
			log.Fatal(err)
		}
		ds.Y = append(ds.Y, y)
	}
	ds.Size = len(ds.X)
	return ds
}

func (ds DataSet) Random() (Vector, float64) {
	i := rand.Intn(ds.Size)
	return ds.X[i], ds.Y[i]
}

func findCols(row []string, target string, features []string) (int, []int) {
	var targetCol int
	var featureCols []int
	for i, word := range row {
		if word == target {
			targetCol = i
		} else {
			for _, f := range features {
				if f == word {
					featureCols = append(featureCols, i)
				}
			}
		}
	}
	return targetCol, featureCols
}

func (ds DataSet) Stats() DataStats {
	var ymean float64
	for _, y := range ds.Y {
		ymean += y
	}
	ymean /= float64(ds.Size)
	var ystd float64
	for _, y := range ds.Y {
		ystd += (y - ymean) * (y - ymean)
	}
	ystd /= float64(ds.Size)
	ystd = math.Sqrt(ystd)

	xmean, xstd := VecStats(ds.X)
	return DataStats{XMean: xmean, XStd: xstd, YMean: ymean, YStd: ystd}
}

func (ds DataSet) Normalise(stats DataStats) DataSet {
	xmean := stats.XMean
	nf := NewVector(len(xmean))
	for i, val := range stats.XStd {
		nf[i] = 1.0 / val
	}
	ymean, ystd := stats.YMean, stats.YStd

	nds := DataSet{
		Size: ds.Size,
		X:    make([]Vector, ds.Size),
		Y:    make([]float64, ds.Size),
	}
	for i, vec := range ds.X {
		nds.X[i] = vec.Add(xmean.ScaMul(-1.0)).Mul(nf)
		nds.Y[i] = (ds.Y[i] - ymean) / ystd
	}
	return nds
}

func (ds DataSet) Split(ratio float64) (DataSet, DataSet) {
	nTest := int(float64(ds.Size) * ratio)
	rand.Shuffle(ds.Size, func(i, j int) {
		ds.X[i], ds.X[j] = ds.X[j], ds.X[i]
		ds.Y[i], ds.Y[j] = ds.Y[j], ds.Y[i]
	})
	testSet := DataSet{X: ds.X[:nTest], Y: ds.Y[:nTest], Size: nTest}
	trainSet := DataSet{X: ds.X[nTest:], Y: ds.Y[nTest:], Size: ds.Size - nTest}
	return trainSet, testSet
}
