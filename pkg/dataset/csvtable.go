package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
)

// type Record []string

type CSVTable struct {
	Header  []string
	Records [][]string
}

func NewCSVTable(path string) (CSVTable, error) {
	tb := CSVTable{}
	fp, err := os.Open(path)
	if err != nil {
		return tb, fmt.Errorf("NewCSVTable.Open: %w", err)
	}
	defer fp.Close()
	rd := csv.NewReader(fp)
	tb.Header, err = rd.Read()
	if err != nil {
		return tb, fmt.Errorf("NewCSVTable.Read: cannot read first line, %w", err)
	}
	tb.Records, err = rd.ReadAll()
	if err != nil {
		return tb, fmt.Errorf("NewCSVTable.ReadAll: cannot read lines, %w", err)
	}
	return tb, nil
}
