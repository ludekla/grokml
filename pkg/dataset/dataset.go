package dataset

import (
	"fmt"
	"log"
	"math/rand"
	"strconv"
)

type dtype interface {
	float64 | string
}

type sample[T dtype] struct {
	dpoint []T
	label  float64
}

type converter[T float64 | string] func(string) T

func AtoF(str string) float64 {
	val, err := strconv.ParseFloat(str, 64)
	if err == nil {
		return val
	}
	switch str {
	case "positive":
		return 1.0
	case "negative":
		return 0.0
	default:
		log.Fatalf("cannot convert %v", str)
		return 0.0
	}
}

func AtoA(str string) string {
	return str
}

type DataSet[T dtype] struct {
	size    int
	header  []string
	samples []sample[T]
}

// fetch data from csv file with specified columns
func NewDataSet[T dtype](rd *CSVReader, conv converter[T]) DataSet[T] {
	ds := DataSet[T]{header: rd.header}
	for {
		row, ok := rd.Read()
		if !ok || len(row) == 0 {
			break
		}
		// datapoint
		dpoint := make([]T, len(row)-1)
		for i, strval := range row[1:] {
			dpoint[i] = conv(strval)
		}
		label := AtoF(row[0])
		ds.samples = append(ds.samples, sample[T]{dpoint, label})
	}
	ds.size = len(ds.samples)
	return ds
}

func (ds DataSet[T]) Size() int {
	return ds.size
}

func (ds DataSet[T]) Header() []string {
	return ds.header
}

func (ds DataSet[T]) DPoints() [][]T {
	dpoints := make([][]T, ds.size)
	for i, sample := range ds.samples {
		dpoints[i] = make([]T, len(sample.dpoint))
		copy(dpoints[i], sample.dpoint)
	}
	return dpoints
}

func (ds DataSet[T]) Labels() []float64 {
	labels := make([]float64, ds.size)
	for i, sample := range ds.samples {
		labels[i] = sample.label
	}
	return labels
}

func (ds DataSet[T]) String() string {
	return fmt.Sprintf(
		"DataSet{size: %d, target: %s, features: %v}",
		ds.size, ds.header[0], ds.header[1:],
	)
}

func (ds DataSet[T]) Random() ([]T, float64) {
	i := rand.Intn(ds.size)
	return ds.samples[i].dpoint, ds.samples[i].label
}

func (ds DataSet[T]) Split(ratio float64) (DataSet[T], DataSet[T]) {
	nTest := int(float64(ds.size) * ratio)
	rand.Shuffle(ds.size, func(i, j int) {
		ds.samples[i], ds.samples[j] = ds.samples[j], ds.samples[i]
	})
	testSet := DataSet[T]{
		samples: ds.samples[:nTest],
		size:    nTest,
	}
	trainSet := DataSet[T]{
		samples: ds.samples[nTest:],
		size:    ds.size - nTest,
	}
	return trainSet, testSet
}
