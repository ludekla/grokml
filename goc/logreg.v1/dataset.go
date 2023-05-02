package logreg

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

type TokenMap map[string]float64

type Example struct {
	tokens TokenMap
	label string
}

type DataSet struct {
	Path string
	Size int
	Header []string
	Examples []Example
}

func NewDataSet(csvfile string) DataSet {
	file, err := os.Open(csvfile)
	if err != nil {
		log.Fatalf("Cannot open CSV file: %v", err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		log.Fatal("cannot read csv line")
	}
	examples := make([]Example, 0, 100)
	recordCh := make(chan []string)
	exampleCh := make(chan Example)
	doneCh := make(chan struct{})
	wg := sync.WaitGroup{}
	go func() {
		defer close(recordCh)
		for {
			rec, err := reader.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal("Cannot read csv line")
			}
			recordCh <- rec
		}
	}()
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rec := range recordCh {
				tm := GetTokenFreqs(rec[0])
				exampleCh <- Example{tokens: tm, label: rec[1]}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(doneCh)
	}()
	for {
		select {
		case example := <-exampleCh:
			examples = append(examples, example)
		case <-doneCh:
			return DataSet{
				Path: csvfile, 
				Size: len(examples), 
				Header: header,
				Examples: examples,
			}		
		}
	}
}

func (ds DataSet) Random() Example {
	i := rand.Intn(ds.Size)
	return ds.Examples[i]
}

func (ds DataSet) Split(ratio float64) (DataSet, DataSet) {
	nTest := int(float64(ds.Size) * ratio)
	rand.Shuffle(ds.Size, func(i, j int) {
		ds.Examples[i], ds.Examples[j] = ds.Examples[j], ds.Examples[i]
	})
	testSet := DataSet{
		Examples: ds.Examples[:nTest], 
		Header: ds.Header, Size: nTest,
	}
	trainSet := DataSet{
		Examples: ds.Examples[nTest:], 
		Header: ds.Header, Size: ds.Size - nTest,
	}
	return trainSet, testSet
}

func GetTokenFreqs(txt string) TokenMap {
	tmap := make(TokenMap)
	tokens := strings.Split(txt, " ")
	var total float64
 	for _, token := range tokens {
		tmap[token] += 1.0
		total++
	}
	for token, count := range tmap {
		tmap[token] = count / total
	} 
	return tmap
}

func (tm TokenMap) Update(tokens TokenMap) {
	for token, _ := range tokens {
		tm[token] = 1.0
	}
}
