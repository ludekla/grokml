package dataset

import (
	"fmt"
	"log"
	"math/rand"
	"strconv"
)

// dtype represents the data type of the dataset samples.
type dtype interface {
	float64 | string
}

// sample holds a pair consisting of a data point and its
// corresponding label.
type sample[T dtype] struct {
	dpoint []T
	label  float64
}

// converter converts a string into the data type T.
type converter[T dtype] func(string) T

// AtoF converts string into T as float64.
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

// AtoA is the trivial map that implements the converter function interface.
func AtoA(str string) string {
	return str
}

// DataSet holds samples of the specified data type T and labels of type float,
// as well as the names of the corresponding attributes.
type DataSet[T dtype] struct {
	size    int
	header  []string
	samples []sample[T]
}

// NewDataSet implements the constructor for DataSet. It needs a CSVReader from
// this package to extract the very columns that the converter function conv
// is applied to.
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

// Size is a getter for the size of the dataset.
func (ds DataSet[T]) Size() int {
	return ds.size
}

// Header is a getter for the column names which represent the attributes' names.
func (ds DataSet[T]) Header() []string {
	return ds.header
}

// DPoints returns a slice of copies of the datapoints to ensure that the original
// data remains untouched when the data is further processed. 
func (ds DataSet[T]) DPoints() [][]T {
	dpoints := make([][]T, ds.size)
	for i, sample := range ds.samples {
		dpoints[i] = make([]T, len(sample.dpoint))
		copy(dpoints[i], sample.dpoint)
	}
	return dpoints
}

// Labels returns copies of the dataset's labels.
func (ds DataSet[T]) Labels() []float64 {
	labels := make([]float64, ds.size)
	for i, sample := range ds.samples {
		labels[i] = sample.label
	}
	return labels
}

// String implements the Stringer interface.
func (ds DataSet[T]) String() string {
	return fmt.Sprintf(
		"DataSet{size: %d, target: %s, features: %v}",
		ds.size, ds.header[0], ds.header[1:],
	)
}

// Random picks a sample at random from its dataset.
func (ds DataSet[T]) Random() ([]T, float64) {
	i := rand.Intn(ds.size)
	return ds.samples[i].dpoint, ds.samples[i].label
}

// Splits returns a random split of the dataset after shuffling its
// samples. This changes the order of the dataset's samples. 
// ratio represents the proportion of the test size. 
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
