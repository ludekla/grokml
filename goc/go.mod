module goc

go 1.17

require (
	goc/tree v0.0.0
	goc/linreg v0.0.0
	goc/logreg v0.1.0
	goc/nbayes v0.0.0
	goc/vaux v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

replace goc/linreg => ./linreg

replace goc/vaux => ./vaux

replace goc/logreg => ./logreg.v1

replace goc/nbayes => ./nbayes

replace goc/tree => ./tree
