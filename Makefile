
fmt:
	go fmt pkg/vector/*
	go fmt pkg/dataset/*
	go fmt pkg/pipeline/*
	go fmt pkg/ch03-linreg/*
	go fmt pkg/ch06-logreg/*
	go fmt pkg/ch08-nbayes/*
	go fmt cmd/ch03-linreg/linreg.go
	go fmt cmd/ch03-linreg/reglin.go
	go fmt cmd/ch06-logreg/sentan.go
	go fmt cmd/ch09-tree/*

vet:
	go vet pkg/vector/*
	go vet pkg/tokens/*
	go vet pkg/dataset/*
	go vet pkg/pipeline/*
	go vet pkg/ch03-linreg/*
	go vet pkg/ch06-logreg/*
	go vet pkg/ch08-nbayes/*
	go vet pkg/ch09-tree/*
	go vet cmd/ch03-linreg/linreg.go
	go vet cmd/ch03-linreg/reglin.go
	go vet cmd/ch06-logreg/sentan.go
	go vet cmd/ch09-tree/tree.go
	go vet cmd/ch09-tree/ada.go
	go vet cmd/ch09-tree/forest.go

test:
	go test pkg/vector/*
	go test pkg/tokens/*
	go test pkg/dataset/*
	go test pkg/pipeline/*
	go test pkg/ch06-logreg/*
	go test pkg/ch08-nbayes/*

run:
	go run cmd/ch03-linreg/linreg.go -t
	go run cmd/ch03-linreg/linreg.go 
	go run cmd/ch03-linreg/reglin.go -t
	go run cmd/ch03-linreg/reglin.go

build:
	go build -o bin/linreg cmd/ch03-linreg/linreg.go
	go build -o bin/reglin cmd/ch03-linreg/reglin.go
	go build -o bin/sentan cmd/ch06-logreg/sentan.go

clean:
	rm bin/* 