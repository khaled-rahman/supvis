#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include "../sample/algorithms.h"

using namespace std;

#define INDEXTYPE int


void helpmessage(){
	printf("\n");
	printf("Usage of BatchLayout tool:\n");
	printf("-input <string>, full path of input file (required).\n");
	printf("-output <string>, directory where output file will be stored.\n");
	printf("-batch <int>, size of minibatch.\n");
	printf("-iter <int>, number of iteration.\n");
	printf("-threads <int>, number of threads, default value is maximum available threads in the machine.\n");
	printf("-h, show help message.\n");

	printf("default: -batch 386 -iter 600 -threads MAX\n\n");
}

void TestAlgorithms(int argc, char *argv[]){
	VALUETYPE energyThreshold = 0.01, bhThreshold = 1.2, gamma = 1.0;
	INDEXTYPE init = 0, batchsize = 384, iterations = 600, numberOfThreads = omp_get_max_threads(), dim = 2, option = 1;
	string inputfile = "", initfile = "", outputfile = "", algoname = "batchlayout", initname = "RAND";
	for(int p = 0; p < argc; p++){
		if(strcmp(argv[p], "-input") == 0){
			inputfile = argv[p+1];
		}
		else if(strcmp(argv[p], "-output") == 0){
			outputfile = argv[p+1];
		}
		else if(strcmp(argv[p], "-batch") == 0){
			batchsize = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-iter") == 0){
			iterations = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-threads") == 0){
			numberOfThreads = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-dim") == 0){
			dim = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-gamma") == 0){
			gamma = atof(argv[p+1]);
		}
		else if(strcmp(argv[p], "-h") == 0){
			helpmessage();
			exit(1);
		}
	}
	algoname = "blembed";
	CSR<INDEXTYPE, VALUETYPE> A_csr;
        SetInputMatricesAsCSR(A_csr, inputfile);
        A_csr.Sorted();
	vector<VALUETYPE> outputvec;
	algorithms algo = algorithms(A_csr, inputfile, outputfile, dim, gamma);
	srand(1);
	outputvec = algo.AlgoBatchLayout(iterations, numberOfThreads, batchsize);
	string avgfile = "Results.txt";
        ofstream output;
       	output.open(avgfile, ofstream::app);
	output << algoname << "\t" << initname << "\t";
       	output << iterations << "\t" << numberOfThreads << "\t" << batchsize << "\t" << dim << "\t";
        output << outputvec[0] << "\t";
	output << endl;
	output.close();
}
int main(int argc, char* argv[]){
	TestAlgorithms(argc, argv);
        return 0;
}
