package dataset

import (
	"encoding/csv"
	"io"
	"log"
	"os"
)

// Helper function to find the column numbers of the columns of interest.
// It finds the index of the strings in needles in the haystack.
func findCols(haystack []string, needles []string) []int {
	cols := make([]int, 0, len(needles))
	for _, title := range needles {
		// go and check for every needle (title) ...
		for i, word := range haystack {
			// ... whether it is in the hackstack
			if title == word {
				cols = append(cols, i)
			}
		}
	}
	return cols
}

// CSVReader extracts rows of the specified columns (cols) from a CSV file,
// represented by a file pointer (file).
type CSVReader struct {
	file   *os.File
	reader *csv.Reader
	header []string
	cols   []int
}

// NewCSVReader constructs a CSVReader.
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

// Read returns the next row of the connected CSV file.
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

// Close closes the underlying file object.
func (rd *CSVReader) Close() error {
	return rd.file.Close()
}
