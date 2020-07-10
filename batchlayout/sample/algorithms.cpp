#include "algorithms.h"


void algorithms::randInit(){
	for(INDEXTYPE i = 0; i < graph.rows; i++){
        	for(INDEXTYPE d = 0; d < DIM; d++){
                 	nCoordinates[i*DIM + d] = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
 			prevCoordinates[i*DIM + d] = nCoordinates[i*DIM + d];			               
		}
       	}		
}

vector<VALUETYPE> algorithms::AlgoBatchLayout(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0.001, start, end, ENERGY, ENERGY0;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInit();
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
        VALUETYPE *prevCoordinates;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
                INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
        }
        while(LOOP < ITERATIONS){
                ENERGY0 = ENERGY;
                ENERGY = 0;
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
                                #pragma forceinline
                                #pragma omp simd
				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[jindex + d]-nCoordinates[iindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
                                        attrc = attrc + 1.0 / attrc;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] += attrc * forceDiff[d];
                                                forceDiff[d] = 0;
                                        }
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                        }
			for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) break;
                                VALUETYPE factor = 0;
                                INDEXTYPE iindex = indices[i] * DIM;
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        factor += prevCoordinates[iindex - baseindex + d] * prevCoordinates[iindex - baseindex + d];
                                }
                                ENERGY += factor;
                                factor = (1.0 * STEP) / sqrt(factor);
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += factor * prevCoordinates[iindex - baseindex + d];
                                        prevCoordinates[iindex - baseindex + d] = 0;
                                }
                        }
                }
                STEP = STEP * 0.999;
                LOOP++;
        }
        end = omp_get_wtime();
        cout << "BatchLayout Wall time required:" << end - start << endl;
        result.push_back(end - start);
        writeToFile("BL"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
        return result;
}

