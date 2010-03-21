//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002-2010
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Detection.h"
#include "dsp/Observation.h"
#include "dsp/Scratch.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"

#include <memory>

#include <string.h>

using namespace std;


__global__ void performPolarimetry (unsigned nchan,const float* p,const float* q,uint64_t ndat,float* S0,float* S2)
{
  unsigned threadIndex = blockIdx.x*blockDim.x + threadIdx.x;

  float p_r;
  float p_i;
  float q_r;
  float q_i;
  float pp;
  float qq;

  if (threadIndex >= ndat*nchan)
    return;

  unsigned k = (threadIndex*2)+(ndat*2*((threadIndex)/ndat));
  //unsigned k = threadIndex*2;
  //unsigned m = threadIndex*2;

  
  //indexLogGPU[m] = &S0[k];
  //indexLogGPU[m+1] = &S0[k+1];
  //indexLogGPUS2[m] = &S2[k];
  //indexLogGPUS2[m+1] = &S2[k+1];
  
  p_r = p[k]; 
  p_i = p[k+1];
  q_r = q[k]; 
  q_i = q[k+1]; 
  
  
  //debugDat[m] = m; 
  //debugDat[m+1] = p_r*p_r + p_i*p_i;
  
  S0[k]   = (p_r * p_r) + (p_i * p_i);  // p * p
  S0[k+1] = (q_r * q_r) + (q_i * q_i);// q * q
  S2[k]   = (p_r * q_r) + (p_i * q_i);  // Re[p * q]
  S2[k+1] = (p_r * q_i) - (p_i * q_r);// Im[p * q]
      
   
}



void polarimetryCUDA (int BlkThread, int Blks, unsigned nchan, unsigned sdet, const float* p, const float* q, uint64_t ndat, unsigned ndim, float* S0, float* S2) 
{
  
  //float** indexLogGPU;
  //cudaMalloc((void**)&indexLogGPU, 2*BlkThread*Blks*sizeof(float*));
  //cudaMemset(indexLogGPU,0,2*BlkThread*Blks*sizeof(float*));

  //float** indexLogGPUS2;
  //cudaMalloc((void**)&indexLogGPUS2, 2*BlkThread*Blks*sizeof(float*));
  //cudaMemset(indexLogGPUS2,0,2*BlkThread*Blks*sizeof(float*));


  //float* debugDat;
  //cudaMalloc((void**)&debugDat,2*BlkThread*Blks*sizeof(float));
  //cudaMemset(debugDat,0,2*BlkThread*Blks*sizeof(float));
  

  //cout << "polarimetryCUDA ndim= " << ndim << " nchan: " << nchan << " sdat " << sdet << " ndat: " << ndat << " BlkThread: " << BlkThread << " Blks: " << Blks << endl;
 
  performPolarimetry<<<BlkThread,Blks>>>(nchan,p,q,ndat,S0,S2);

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "FAIL performPolarimetry: " << cudaGetErrorString (error) << endl;
  cudaThreadSynchronize();
  

  //float* debugDatCPU;
  //debugDatCPU = (float*) malloc(2*BlkThread*Blks*sizeof(float));
  //cudaMemcpy(debugDatCPU,S0,2*BlkThread*Blks*sizeof(float),cudaMemcpyDeviceToHost);
  /*

  float** indexLogCPU;
  indexLogCPU = (float**) malloc(2*BlkThread*Blks*sizeof(float*));
  
  for (unsigned i=0;i<BlkThread*Blks*2;i++)
    indexLogCPU[i] = 0;

  cudaMemcpy(indexLogCPU,indexLogGPU,2*BlkThread*Blks*sizeof(float*),cudaMemcpyDeviceToHost);

  float** indexLogCPUS2;
  indexLogCPUS2 = (float**) malloc(2*BlkThread*Blks*sizeof(float*));
  
  for (unsigned i=0;i<BlkThread*Blks*2;i++)
    indexLogCPUS2[i] = 0;

    cudaMemcpy(indexLogCPUS2,indexLogGPUS2,2*BlkThread*Blks*sizeof(float*),cudaMemcpyDeviceToHost);
    // if (ndat == 1362) {*/



  //    for (unsigned i=0;i<10;i++)
  //    cout << "debugDat: " << debugDatCPU[i] << endl;


    //cout << "ndat: " << ndat << " indexLogCPU: " << indexLogCPU[i] << " indexLogCPUS2: " << indexLogCPUS2[i] << " debugDat: " << debugDatCPU[i] << endl;
	  //}
    //cudaFree(indexLogGPU);
    //cudaFree(indexLogGPUS2);
    //cudaFree(debugDat);
  //free(indexLogCPUS2);
  //free(indexLogCPU);
  //free(debugDatCPU);
  //cout << "finished" << endl;
  
}



