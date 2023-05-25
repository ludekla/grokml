package dataset

import (
	"encoding/csv"
	"io"
	"log"
	"os"
)

func findCols(row []string, header []string) []int {
	cols := make([]int, 0, len(header))
	for _, title := range header {
		for i, word := range row {
			if title == word {
				cols = append(cols, i)
			}
		}
	}
	return cols
}

type CSVReader struct {
	file   *os.File
	reader *csv.Reader
	header []string
	cols   []int
}

func NewCSVReader(path string, header ...string) *CSVReader {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(file)
	rec, err := reader.Read()
	if err != nil {
		log.Fatal(err)
	}
	cols := findCols(rec, header)
	return &CSVReader{file, reader, header, cols}
}

func (rd *CSVReader) Read() ([]string, bool) {
	rec, err := rd.reader.Read()
	if err == io.EOF {
		rd.file.Close()
		return nil, false
	} else if err != nil {
		rd.file.Close()
		log.Fatal(err)
	}
	row := make([]string, len(rd.cols))
	for j, idx := range rd.cols {
		row[j] = rec[idx]
	}
	return row, true
}

func (rd *CSVReader) Close() error {
	return rd.file.Close()
}
