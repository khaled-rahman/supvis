#ifndef _ALGORITHMS_H_
#define _ALGORITHMS_H_

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <map>
#include <new>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <stack>
#include <string>
#include <sstream>
#include <random>
#include <limits>
#include <cassert>
#include <parallel/algorithm>
#include <parallel/numeric>
//#include <parallel/random_shuffle>

#include "CSR.h"

#include "utility.h"

#include "commonutility.h"


using namespace std;

#define VALUETYPE float
#define INDEXTYPE int
#define MAXBOUND 6
#define MAXMIN 3.0
#define t 0.99
#define PI 3.14159265358979323846
static int PROGRESS = 0;

//#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv = Coordinate<VALUETYPE>(0.0, 0.0))

class algorithms{
	public:
		CSR<INDEXTYPE, VALUETYPE> graph;
		VALUETYPE *nCoordinates, *prevCoordinates;
		VALUETYPE GAMMA = 1.0;
		INDEXTYPE DIM;
		string filename;
		string outputdir;
	public:
	algorithms(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int dim, VALUETYPE gm):nCoordinates(new VALUETYPE[A_csr.rows * dim]),prevCoordinates(new VALUETYPE[A_csr.rows * dim]){
		graph.make_empty();
		graph = A_csr;
		outputdir = outputd;
		filename = input;
		DIM = dim;
		GAMMA = gm;
		//printCSR(A_csr);
		//static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[A_csr.rows])));
	}
	~algorithms(){
		delete[] nCoordinates;
		delete[] prevCoordinates;
	}

	/********************/
	void randInit();
	vector<VALUETYPE> AlgoBatchLayout(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	/********************/

	void print(){
		for(INDEXTYPE i = 0; i < graph.rows * DIM; i += DIM){
			for(INDEXTYPE d = 0; d < DIM; d++){
                		cout << "Node:" << i << ", X:" << nCoordinates[i+d] << ", Y:" << nCoordinates[i+d+1] << endl;
        		}
		}
		cout << endl;
	}
	void writeToFile(string f){
		stringstream  data(filename);
    		string lasttok;
    		while(getline(data,lasttok,'/'));
		filename = outputdir + lasttok + f + ".txt";
		ofstream output;
		output.open(filename);
		cout << "Creating output file in following directory:" << filename << endl;
		for(INDEXTYPE i = 0; i < graph.rows * DIM; i += DIM){
			output << int(i / DIM) << "\t";
			for(INDEXTYPE d = 0; d < this->DIM; d++){
				output << nCoordinates[i+d] <<"\t";
			}
			output << endl;
		}
		output.close();
	}
};
#endif
