package tokens

import "encoding/json"

// NonScaler is a trivial dummy implementation of the Scaler interface.
// The sole purpose is to enable the set-up of pipelines that need no scaling
// operation.
type NonScaler struct{}

func NewNonScaler() *NonScaler {
	return &NonScaler{}
}

func (sc *NonScaler) Fit(tmaps []TokenMap) {}

func (sc NonScaler) Transform(tmaps []TokenMap) []TokenMap {
	return tmaps
}

// Marshal and Unmarshal implement the JSONable interface (persist pkg).
func (sc NonScaler) Marshal() ([]byte, error) {
	return json.MarshalIndent(sc, "", "    ")
}

func (sc *NonScaler) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, sc)
}
