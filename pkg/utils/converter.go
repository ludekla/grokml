package utils

import (
	"fmt"
	"strconv"
)

type DataPoint interface {
	Vector | string
}

type Label interface {
	int | float64
}

type Converter[D DataPoint, L Label] interface {
	Row2DataPoint(row []string) (D, error)
	Str2Label(target string) (L, error)
}

type ConvertFloat struct{}

func (cf ConvertFloat) Row2DataPoint(row []string) (Vector, error) {
	vec := NewVector(len(row))
	for i, val := range row {
		fval, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return vec, err
		}
		vec[i] = fval
	}
	return vec, nil
}

func (cf ConvertFloat) Str2Label(str string) (float64, error) {
	return strconv.ParseFloat(str, 64)
}

type ConvertText struct{}

func (ct ConvertText) Row2DataPoint(row []string) (string, error) {
	if len(row) == 0 {
		return "", fmt.Errorf("ConvertText: no data.")
	}
	return row[0], nil
}

func (cf ConvertText) Str2Label(str string) (float64, error) {
	switch str {
	case "positive", "1":
		return 1.0, nil
	case "negative", "0":
		return 0.0, nil
	default:
		return 0.0, fmt.Errorf("ConvertText: cannot interpret label %s", str)
	}
}
