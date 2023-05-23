package utils

import (
	"fmt"
	"log"
	"math/rand"
)

type DataSet[D DataPoint, L Label] struct {
	size    int
	header  []string
	dpoints []D
	labels  []L
}

// fetch data from csv file with specified columns
func NewDataSet[D DataPoint, L Label](rd *CSVReader, conv Converter[D, L]) DataSet[D, L] {
	ds := DataSet[D, L]{header: rd.header}
	for {
		row, ok := rd.Read()
		if !ok {
			break
		}
		// label
		label, err := conv.Str2Label(row[0])
		if err != nil {
			rd.Close()
			log.Fatal(err)
		}
		ds.labels = append(ds.labels, label)
		// datapoint
		dpoint, err := conv.Row2DataPoint(row[1:])
		if err != nil {
			rd.Close()
			log.Fatal(err)
		}
		ds.dpoints = append(ds.dpoints, dpoint)
	}
	ds.size = len(ds.dpoints)
	return ds
}

func (ds DataSet[D, L]) Size() int {
	return ds.size
}

func (ds DataSet[D, L]) Header() []string {
	return ds.header
}

func (ds DataSet[D, L]) X() []D {
	return ds.dpoints
}

func (ds DataSet[D, L]) Y() []L {
	return ds.labels
}

func (ds DataSet[D, L]) String() string {
	return fmt.Sprintf(
		"DataSet{size: %d, target: %s, features: %v}",
		ds.size, ds.header[0], ds.header[1:],
	)
}

func (ds DataSet[D, L]) Random() (D, L) {
	i := rand.Intn(ds.size)
	return ds.dpoints[i], ds.labels[i]
}

func (ds DataSet[D, L]) Split(ratio float64) (DataSet[D, L], DataSet[D, L]) {
	nTest := int(float64(ds.size) * ratio)
	rand.Shuffle(ds.size, func(i, j int) {
		ds.dpoints[i], ds.dpoints[j] = ds.dpoints[j], ds.dpoints[i]
		ds.labels[i], ds.labels[j] = ds.labels[j], ds.labels[i]
	})
	testSet := DataSet[D, L]{
		dpoints: ds.dpoints[:nTest],
		labels:  ds.labels[:nTest],
		size:    nTest,
	}
	trainSet := DataSet[D, L]{
		dpoints: ds.dpoints[nTest:],
		labels:  ds.labels[nTest:],
		size:    ds.size - nTest,
	}
	return trainSet, testSet
}
