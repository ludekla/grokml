package tree

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
)

type Example struct {
	features []float64
	target   float64
}

type DataSet struct {
	Size     int
	Header   []string
	Examples []Example
}

func (x Example) Features() []float64 {
	return x.features
}

func NewDataSet(csvfile string) DataSet {
	file, err := os.Open(csvfile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	rd := csv.NewReader(file)
	header, err := rd.Read()
	if err != nil {
		log.Fatalln("Cannot read header of CSV file")
	}
	recordCh := make(chan []string)
	exampleCh := make(chan Example)
	doneCh := make(chan struct{})
	go func() {
		defer close(recordCh)
		for {
			rec, err := rd.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatalln("Cannot read CSV line")
			}
			recordCh <- rec
		}
	}()
	wg := sync.WaitGroup{}
	nWorkers := runtime.NumCPU()
	n := len(header)
	go func() {
		wg.Wait()
		close(doneCh)
	}()
	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			for rec := range recordCh {
				vals := make([]float64, n)
				for j, val := range rec {
					vals[j], err = strconv.ParseFloat(val, 64)
					if err != nil {
						log.Fatal(err)
					}
				}
				exampleCh <- Example{features: vals[:n-1], target: vals[n-1]}
			}
		}()
	}
	examples := make([]Example, 0, 100)
	for {
		select {
		case example := <-exampleCh:
			examples = append(examples, example)
		case <-doneCh:
			return DataSet{
				Size:     len(examples),
				Header:   header,
				Examples: examples,
			}
		}
	}
}

func (ds DataSet) Split(testRatio float64) (DataSet, DataSet) {
	n := int(float64(ds.Size) * testRatio)
	rand.Shuffle(ds.Size, func(i, j int) {
		ds.Examples[i], ds.Examples[j] = ds.Examples[j], ds.Examples[i]
	})
	testSet := DataSet{
		Size:     n,
		Header:   ds.Header,
		Examples: ds.Examples[:n],
	}
	trainSet := DataSet{
		Size:     ds.Size - n,
		Header:   ds.Header,
		Examples: ds.Examples[n:],
	}
	return trainSet, testSet
}

func (ds DataSet) Random() Example {
	i := rand.Intn(ds.Size)
	return ds.Examples[i]
}

func (ds DataSet) Copy() DataSet {
	dset := DataSet{Size: ds.Size, Examples: make([]Example, ds.Size)}
	for i, example := range ds.Examples {
		dset.Examples[i] = Example{
			features: cpfloats(example.features),
			target:   example.target,
		}
	}
	return dset
}

func cpfloats(fs []float64) []float64 {
	newfs := make([]float64, len(fs))
	for i, f := range fs {
		newfs[i] = f
	}
	return newfs
}
