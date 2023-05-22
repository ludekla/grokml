package ch08

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
)

type Example struct {
	Text string
	Spam bool
}

type DataSet struct {
	Size   int
	Header []string
	Data   []Example
}

func NewDataSet(path string) DataSet {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		log.Fatal(err)
	}
	rawCh := make(chan []string)
	go func() {
		defer close(rawCh)
		for {
			rec, err := reader.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal("Cannot read csv line")
			}
			rawCh <- rec
		}
	}()
	exampleCh := make(chan Example)
	wg := sync.WaitGroup{}
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rec := range rawCh {
				txt := strings.ToLower(rec[0])
				var spam bool
				if rec[1] == "1" {
					spam = true
				}
				exampleCh <- Example{Text: txt, Spam: spam}
			}
		}()
	}
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()
	examples := make([]Example, 0, 100)
	for {
		select {
		case example := <-exampleCh:
			examples = append(examples, example)
		case <-doneCh:
			return DataSet{
				Size:   len(examples),
				Header: header,
				Data:   examples,
			}
		}
	}
}

func (ds DataSet) Split(testRatio float64) (DataSet, DataSet) {
	rand.Shuffle(ds.Size, func(i, j int) {
		ds.Data[i], ds.Data[j] = ds.Data[j], ds.Data[i]
	})
	nTest := int(testRatio * float64(ds.Size))
	testSet := DataSet{Size: nTest, Header: ds.Header, Data: ds.Data[:nTest]}
	trainSet := DataSet{Size: ds.Size - nTest, Header: ds.Header, Data: ds.Data[nTest:]}
	return trainSet, testSet
}
