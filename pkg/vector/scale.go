// Implements vector scaling.
package vector

import "encoding/json"

// Scaler implements the Scaler interface (pipeline pkg)
type Scaler struct {
	Mean Vector `json:"mean"`
	Std  Vector `json:"std"`
}

// NewScaler is the factory function for the Scaler struct.
func NewScaler() *Scaler {
	return &Scaler{}
}

// Fit computes both the vector mean and standard deviation, setting the
// parameters of the Scaler. This is the preparation for the scaling procedure.
// It also implements the Fit method of the Scale interface.
func (sc *Scaler) Fit(vecs []Vector) {
	mean, std := VectorStats(vecs)
	sc.Mean = mean
	sc.Std = std
}

// Transform implements the Transform method of the Scaler interface. It normalises
// the given vectors by substracting the mean and scaling by its standard deviation.
func (sc Scaler) Transform(vecs []Vector) []Vector {
	mean, std := sc.Mean, sc.Std
	outvecs := make([]Vector, len(vecs))
	for i, vec := range vecs {
		outvecs[i] = vec.Add(mean.ScaMul(-1.0)).Div(std)
	}
	return outvecs
}

// Marshal and Unmarshal implement the JSONable interface (persist pkg).
func (sc Scaler) Marshal() ([]byte, error) {
	return json.MarshalIndent(sc, "", "    ")
}

func (sc *Scaler) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, sc)
}
