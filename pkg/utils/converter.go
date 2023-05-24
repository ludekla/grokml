package utils

import (
	"fmt"
	"strconv"
)

func ToVector(row []string) (Vector, error) {
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

func ToStr(row []string) (string, error) {
	if len(row) == 0 {
		return "", fmt.Errorf("ConvertText: no data.")
	}
	return row[0], nil
}

func Str2Label(str string) (float64, error) {
	val, err := strconv.ParseFloat(str, 64)
	if err == nil {
		return val, nil
	}
	switch str {
	case "positive":
		return 1.0, nil
	case "negative":
		return 0.0, nil
	default:
		return 0.0, fmt.Errorf("ConvertText: cannot interpret label %s", str)
	}
}
