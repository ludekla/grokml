## GROKML

# Introduction

This repo is an exercise in implementing machine learning algorithms in Go, where
we have tried our hand at *generics*, a relatively recent addition to the language. 

In particular, the challenge was to implement the *pipeline* concept, as it is
known to Pythonistas using the *scikit-learn* module. 

The project structure roughly follows the book 

*Luis G. Serrano: Grokking Machine Learning, Manning Publications Co. (2021)*

except for the part on neural networks. We exclude neural networks here. They
are quite complex and require extra care. When it comes to recurrent neural networks,
especially LSTMs, they tend - at least in my experience - to overwhelm the garbage collector.
I will have to do further research to figure out why that happened and whether I could have
done better. 

# Usage

Users may just clone the repo and play with the code. People who are trying to *grok* ML 
algorithms or are even working through the book will find it useful to look through the 
code, it should be structured reasonably well. 

Always using Python will not have the same learning effect as coding ML algos in a compiled
language such as Go. It requires you to go the whole hog.  

The drivers in *cmd* and the main file are all command-line apps with only one flag: *-t*.

Raise it, and the algorithm will be trained and the state of the corresponding model be saved
in the form of a JSON in the model cache folder named *models*. This enables you to look at it
and inspect it.

If you do not use the flag, the trained and cached model will be loaded. If the required JSON
is not available because you have deleted it, you will cause Go panic.

The *Makefile* will vet the code and perform tests.

*data* contains some standard datasets as CSV files that can be found anywhere on the net. 
The code draws on them.