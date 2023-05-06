package utils

import (
	"encoding/csv"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"fmt"
)

type DataSet struct {
	path     string
	size     int
	features []string
	target   string
	x        []Vector
	y        []float64
}

type DataStats struct {
	XMean Vector  `json:"xmean"`
	XStd  Vector  `json:"xstd"`
	YMean float64 `json:"ymean"`
	YStd  float64 `json:"ystd"`
}

// fetch data from csv file with specified columns
func NewDataSet(csvfile string, target string, features []string) DataSet {
	ds := DataSet{path: csvfile}
	file, err := os.Open(csvfile)
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
		ds.x = append(ds.x, row)
		y, err := strconv.ParseFloat(rec[targetCol], 64)
		if err != nil {
			log.Fatal(err)
		}
		ds.y = append(ds.y, y)
	}
	ds.size = len(ds.x)
	ds.features = features
	ds.target = target
	return ds
}

func (ds DataSet) Size() int {
	return ds.size
}

func (ds DataSet) Features() []string {
	return ds.features
}

func (ds DataSet) X() []Vector {
	return ds.x
}

func (ds DataSet) Target() string {
	return ds.target
}

func (ds DataSet) Y() []float64 {
	return ds.y
}

func (ds DataSet) String() string {
	return fmt.Sprintf(
		"DataSet{size: %d, target: %s, features: %v}", 
		ds.size, ds.target, ds.features,
	)
}

func (ds DataSet) Random() (Vector, float64) {
	i := rand.Intn(ds.size)
	return ds.x[i], ds.y[i]
}

func findCols(row []string, target string, features []string) (int, []int) {
	var targetCol int
	featureCols := make([]int, 0, len(features))
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
	for _, y := range ds.y {
		ymean += y
	}
	ymean /= float64(ds.size)
	var ystd float64
	for _, y := range ds.y {
		ystd += (y - ymean) * (y - ymean)
	}
	ystd /= float64(ds.size)
	ystd = math.Sqrt(ystd)

	xmean, xstd := VecStats(ds.x)
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
		size: ds.size,
		x:    make([]Vector, ds.size),
		y:    make([]float64, ds.size),
	}
	for i, vec := range ds.x {
		nds.x[i] = vec.Add(xmean.ScaMul(-1.0)).Mul(nf)
		nds.y[i] = (ds.y[i] - ymean) / ystd
	}
	return nds
}

func (ds DataSet) Split(ratio float64) (DataSet, DataSet) {
	nTest := int(float64(ds.size) * ratio)
	rand.Shuffle(ds.size, func(i, j int) {
		ds.x[i], ds.x[j] = ds.x[j], ds.x[i]
		ds.y[i], ds.y[j] = ds.y[j], ds.y[i]
	})
	testSet := DataSet{x: ds.x[:nTest], y: ds.y[:nTest], size: nTest}
	trainSet := DataSet{x: ds.x[nTest:], y: ds.y[nTest:], size: ds.size - nTest}
	return trainSet, testSet
}