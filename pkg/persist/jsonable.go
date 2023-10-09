package persist

import (
	"fmt"
	"os"
)

// JSONable is an interface for dumping and loading trained models.
type JSONable interface {
	Marshal() ([]byte, error)
	Unmarshal(bs []byte) error
}

// Dump saves the trained model parameters to a JSON file.
// The model must conform with the JSONable interface.
func Dump(jn JSONable, filepath string) error {
	asBytes, err := jn.Marshal()
	if err != nil {
		return fmt.Errorf("cannot marshal %v into JSON bytes", jn)
	}
	err = os.WriteFile(filepath, asBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write JSON bytes into file %s", filepath)
	}
	return nil
}

// Load takes the model parameters from a JSONfile and lets the model
// fill its struct fields.
func Load(jn JSONable, filepath string) error {
	asBytes, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("cannot read from file %s: %v", filepath, err)
	}
	err = jn.Unmarshal(asBytes)
	if err != nil {
		return fmt.Errorf("cannot unmarshal JSON bytes: %v", err)
	}
	return nil
}
