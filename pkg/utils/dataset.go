package utils

import (
	"fmt"
	"log"
	"math/rand"
)

type DataPoint interface {
	string | Vector | TokenMap
}

type DataSet[D DataPoint] struct {
	size    int
	header  []string
	dpoints []D
	labels  []float64
}

type Converter[D DataPoint] func(row []string) (D, error)

// fetch data from csv file with specified columns
func NewDataSet[D DataPoint](rd *CSVReader, conv Converter[D]) DataSet[D] {
	ds := DataSet[D]{header: rd.header}
	for {
		row, ok := rd.Read()
		if !ok {
			break
		}
		// label
		label, err := Str2Label(row[0])
		if err != nil {
			rd.Close()
			log.Fatal(err)
		}
		ds.labels = append(ds.labels, label)
		// datapoint
		dpoint, err := conv(row[1:])
		if err != nil {
			rd.Close()
			log.Fatal(err)
		}
		ds.dpoints = append(ds.dpoints, dpoint)
	}
	ds.size = len(ds.dpoints)
	return ds
}

func (ds DataSet[D]) Size() int {
	return ds.size
}

func (ds DataSet[D]) Header() []string {
	return ds.header
}

func (ds DataSet[D]) X() []D {
	return ds.dpoints
}

func (ds DataSet[D]) Y() []float64 {
	return ds.labels
}

func (ds DataSet[D]) String() string {
	return fmt.Sprintf(
		"DataSet{size: %d, target: %s, features: %v}",
		ds.size, ds.header[0], ds.header[1:],
	)
}

func (ds DataSet[D]) Random() (D, float64) {
	i := rand.Intn(ds.size)
	return ds.dpoints[i], ds.labels[i]
}

func (ds DataSet[D]) Split(ratio float64) (DataSet[D], DataSet[D]) {
	nTest := int(float64(ds.size) * ratio)
	rand.Shuffle(ds.size, func(i, j int) {
		ds.dpoints[i], ds.dpoints[j] = ds.dpoints[j], ds.dpoints[i]
		ds.labels[i], ds.labels[j] = ds.labels[j], ds.labels[i]
	})
	testSet := DataSet[D]{
		dpoints: ds.dpoints[:nTest],
		labels:  ds.labels[:nTest],
		size:    nTest,
	}
	trainSet := DataSet[D]{
		dpoints: ds.dpoints[nTest:],
		labels:  ds.labels[nTest:],
		size:    ds.size - nTest,
	}
	return trainSet, testSet
}
