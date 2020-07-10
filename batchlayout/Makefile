BatchLayout = .
SAMPLE = ./sample
UNITTEST = ./Test
BIN = ./bin
INCDIR = -I$(SAMPLE) -I$(UNITTEST)

COMPILER = g++

FLAGS = -g -fomit-frame-pointer -fopenmp -O3 -std=c++11 -DCPP

all: batchlayout

algorithms.o:	$(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h $(SAMPLE)/IO.h $(SAMPLE)/CSR.h $(SAMPLE)/CSC.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/algorithms.o $(SAMPLE)/algorithms.cpp

batchlayout.o:  $(UNITTEST)/BatchLayoutEmbed.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/batchlayout.o $(UNITTEST)/BatchLayoutEmbed.cpp

batchlayout:    algorithms.o batchlayout.o
		$(COMPILER) $(INCDIR) $(FLAGS) -o $(BIN)/BatchLayoutEmbed $(BIN)/algorithms.o $(BIN)/batchlayout.o

clean:
	rm -rf ./bin/*
