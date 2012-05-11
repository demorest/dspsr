//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/CyclicFoldEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>
#include <fstream>

using namespace std;

CUDA::CyclicFoldEngineCUDA::CyclicFoldEngineCUDA (cudaStream_t _stream)
{
  lagbinplan = NULL;
  d_binplan = NULL;
  d_lagdata = NULL;
  current_turn = 0;


  // no data on either the host or device
  synchronized = true;

  stream = _stream;
}

CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA ()
{
  
  if (lagbinplan) {
    cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing lagbinplan" <<endl;
    delete [] lagbinplan;
  }
  if (d_binplan) {
    cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing d_binplan" <<endl;
    cudaFree(d_binplan);
  }
  if (d_lagdata) {
    cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing d_lagdata" <<endl;
    cudaFree(d_lagdata);
  }
  cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA finished" <<endl;
}

void CUDA::CyclicFoldEngineCUDA::synch (dsp::PhaseSeries *out) try
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch this=" << this << endl;

  if (synchronized)
    return;

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch output=" << output << endl;

  // transfer lag data from GPU

  cudaError error;
  if (stream)
    error = cudaMemcpyAsync (lagdata,d_lagdata,lagdata_size*sizeof(float),cudaMemcpyDeviceToHost,stream);
  else
    error = cudaMemcpy (lagdata,d_lagdata,lagdata_size*sizeof(float),cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::sync",
                 "cudaMemcpy%s %s", 
                 stream?"Async":"", cudaGetErrorString (error));

  // Call usual synch() to do transform
  dsp::CyclicFoldEngine::synch(out);

  synchronized = true;
}
catch (Error& error)
{
  throw error += "CUDA::CyclicFoldEngineCUDA::synch";
}

void CUDA::CyclicFoldEngineCUDA::set_ndat (uint64_t _ndat, uint64_t _idat_start)
{
	// ndat is idat_end - idat_start
	// binplan_size is _ndat
  setup();

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat ndat=" << _ndat << endl;

  if (_ndat > binplan_size) {

    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_ndat alloc binplan" << endl;

    if (lagbinplan) {
      delete [] lagbinplan;
    }
    lagbinplan = new bin [_ndat*nlag];


    binplan_size = _ndat;
  }
  memset(lagbinplan, 0 , sizeof(bin)*_ndat*nlag);
  current_turn = 0;
  last_ibin = 0;
  ndat_fold = _ndat;
  idat_start = _idat_start;

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat "
      << "nlag=" << nlag << " "
      << "nbin=" << nbin << " "
      << "npol=" << npol_out << " "
      << "nchan=" << nchan << endl;

  uint64_t _lagdata_size = nlag * nbin * npol_out * ndim * nchan;

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat lagdata_size=" << _lagdata_size << endl;

  if (_lagdata_size > lagdata_size) {
    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_ndat alloc lagdata" << endl;
    if (lagdata) delete [] lagdata;
    lagdata = new float [_lagdata_size];
    lagdata_size = _lagdata_size;
    memset(lagdata, 0, sizeof(float)*lagdata_size);
    
    if (d_lagdata) cudaFree(d_lagdata);
    cudaMalloc((void**)&d_lagdata, lagdata_size * sizeof(float));
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
    
  }


}

void CUDA::CyclicFoldEngineCUDA::set_bin (uint64_t idat, double d_ibin, 
        double bins_per_sample)
{
	// idat ranges from idat_start to idat_start + binplansize
  unsigned ibin;
  int ilag;
  if ((last_ibin == nbin-1) && (last_ibin != int(d_ibin))) {
	  current_turn++;
  }
  for (ilag=0;ilag<nlag;ilag++) {
    ibin = unsigned(d_ibin + (((double)ilag)*bins_per_sample)/2.0); //half sample spacing
    unsigned ribin = ibin;
    unsigned planidx = current_turn*nbin*nlag + ibin*nlag + ilag;
    ibin = ibin % nbin; // ibin is wrapped phase
    if (lagbinplan[planidx].hits == 0) {
      lagbinplan[planidx].offset = idat;
      lagbinplan[planidx].ibin = ibin;
      lagbinplan[planidx].hits += 1;
    } 
    else {
      lagbinplan[planidx].hits += 1;
    }
//    cerr << "net:" << ribin + current_turn*nbin << " turn:" << current_turn << " ilag:" << ilag << " idx:" << planidx
//    		<< " hits:" << lagbinplan[planidx].hits << " offs:" << lagbinplan[planidx].offset << " idat:" << idat
//    		<< " ibin:" << ibin << " rbin:" << ribin << endl;
  }
  ndat_fold ++;
  last_ibin = int(d_ibin);
}
  


void CUDA::CyclicFoldEngineCUDA::zero ()
{
  dsp::CyclicFoldEngine::zero();
  if (d_lagdata && lagdata_size>0) {
	  cerr << "CUDA::CyclicFoldEngineCUDA::zero: zeroing lagdata on gpu" << endl;
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
  }
}

void CUDA::CyclicFoldEngineCUDA::send_binplan ()
{

	/*
	 * current_turn is the highest number of turns that we needed in the set_bin stage
	 * so the total size of the binplan should be current_turn turns of nbin bins and nlag lags
	 * so we will update binplan_size accordingly
	 */
	uint64_t orig_size = binplan_size;
	binplan_size = (current_turn + 1) * nbin; // add one turn just for good measure. There should be zero hits in it
  uint64_t mem_size = binplan_size * nlag * sizeof(bin);

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan ndat=" << ndat_fold 
         << "  Allocating on device mem_size " << mem_size
         << " binplan_size=" << binplan_size
         << " nlag=" << nlag
         << " sizeof(bin)=" << sizeof(bin)
         << " current_turn=" << current_turn
         << " orig_size=" << orig_size
         << endl;

  cudaError error;
  
  if (d_binplan == NULL) {
	  cerr << "no binplan yet allocated" << endl;
    error = cudaMalloc ((void **)&(d_binplan),mem_size); // TODO: is this the right way to do this cudaMalloc call? taken from example online: http://stackoverflow.com/questions/6515303/cuda-cudamalloc
    if (error != cudaSuccess)
        throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
                     "cudaMalloc orig %s %s",
                     stream?"Async":"", cudaGetErrorString (error));
  } else {
	  // original plan was to check if binplan_size < orig_size so as to avoid extraneous free/malloc, but it
	  // seems that binplan_size gets reset each time before this funciton is called.
	  cerr << "orig_size=" << orig_size << "< binplansize=" << binplan_size << "so freeing.." << endl;
	  error =cudaFree(d_binplan);
	  if (error != cudaSuccess)
		  throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
					   "cudaFree %s %s",
					   stream?"Async":"", cudaGetErrorString (error));
	  cerr << "realocating..." << endl;
	  error = cudaMalloc ((void **)&(d_binplan),mem_size);
	  if (error != cudaSuccess)
		  throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
					   "cudaMalloc new %s %s",
					   stream?"Async":"", cudaGetErrorString (error));
  }

/*  for (int k=binplan_size*nlag-nlag*16; k > 0; k -= nlag*nbin)
  {
	  if (lagbinplan[k].hits > 0){
		  cerr << "Found some hits at k = " << k << " = " << (k/(nlag*nbin)) << endl;
		  cerr << "current turn=" << current_turn << endl;
		  break;
	  }
  }*/

/*  ofstream fbin;
  fbin.open("cudabinplan.dat", ios::binary | ios::app);
  fbin.write((char *)(lagbinplan),mem_size);
  cerr << "done, dumping cudabinplan, closing files" << endl;
  fbin.close();
*/
  cerr << "copying: stream=" << stream << " d_binplan=" << d_binplan << " mem_size=" << mem_size <<
		  " lagbinplan=" << lagbinplan << endl;
  if (stream)
    error = cudaMemcpyAsync (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice,stream);
  else
    error = cudaMemcpy (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
                 "cudaMemcpy%s %s", 
                 stream?"Async":"", cudaGetErrorString (error));
}

void CUDA::CyclicFoldEngineCUDA::get_lagdata ()
{
  size_t lagdata_bytes = lagdata_size * sizeof(float);
  cudaError error;
  if (stream) 
    error = cudaMemcpyAsync (lagdata, d_lagdata, lagdata_bytes,
        cudaMemcpyDeviceToHost, stream);
  else
    error = cudaMemcpy (lagdata, d_lagdata, lagdata_bytes,
        cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::get_lagdata",
                 "cudaMemcpy%s %s", 
                 stream?"Async":"", cudaGetErrorString (error));
}

/* 
 *  CUDA Kernels
 *
 */
// threadIdx.x -> ilaga    blockDim.x
// threadIdx.y -> pol
// threadIdx.z -> not used
// blockIdx.x -> ilagb
// blockIdx.y -> ibin
// blockIdx.z = ichan

// data is in FPT order, so chunks of time for a given pol and frequency
// in_span gives size of one time chunk for a given freq and pol
__global__ void cycFoldIndPol (const float* in_base,
                unsigned in_span,
                float* out_base,
                unsigned binplan_size,
                unsigned nlag,
                CUDA::bin* binplan)
{
  unsigned ilaga = threadIdx.x;
  unsigned nlaga = blockDim.x;
  unsigned ilagb = blockIdx.x;
  unsigned ibin = blockIdx.y;
  unsigned ichan = blockIdx.z;
  unsigned ipol = threadIdx.y;
  unsigned npol = blockDim.y;
  unsigned nbin = gridDim.y;
  unsigned nchan = gridDim.z;
  unsigned ilag = ilagb*nlaga + ilaga;
  if (ilag >= nlag){
	  return;
  }
  unsigned planidx = nlag*ibin+ilag;
  const unsigned ndim = 2; // always complex data assumed

  if (planidx >= binplan_size) {
    return;
  }
  
  in_base  += in_span  * (ichan*npol + ipol);
//  out_base += out_span * (ichan*npol + ipol);
  out_base += ndim*(ibin*npol*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag 
    + ilag);
  
  unsigned bpstep = nlag*nbin; // step size to get to the next rotation for a given lag and bin

  float2 total = make_float2(0.0,0.0);

  for (; planidx < binplan_size; planidx += bpstep)
  {
    const float* input = in_base + binplan[planidx].offset * ndim;
    const float* input2 = in_base + (binplan[planidx].offset + ilag) * ndim;
    const float2* a = (const float2*)(input);
    const float2* b = (const float2*)(input2);    

    for (unsigned i=0; i < binplan[planidx].hits; i++){
      total.x += a[i].x*b[i].x + a[i].y*b[i].y;
      total.y += a[i].y*b[i].x - a[i].x*b[i].y;
    }
  }

  out_base[0] += total.x;
  out_base[1] += total.y;
} 

void check_error (const char*);


void CUDA::CyclicFoldEngineCUDA::fold ()
{

  // TODO state/etc checks

  cerr << "In CyclicFoldEngineCUDA::fold" << endl;
  setup ();
  send_binplan ();
  const unsigned THREADS_PER_BLOCK = 1024;
  unsigned nlaga,nlagb;
  // if nlag*npol < THREADS_PER_BLOCK then nlaga = nlag, nlagb = 1
  // else nlaga = THREADS_PER_BLOCK/npol, nlagb = nlag/nlaga + 1
  if (nlag*npol > THREADS_PER_BLOCK) {
	  nlaga = THREADS_PER_BLOCK/npol;
	  nlagb = nlag/nlaga + 1;
  }
  else {
	  nlagb = 1;
	  nlaga = nlag;
  }

  dim3 blockDim (nlaga, npol, 1);
  dim3 gridDim (nlagb, nbin, nchan);
  cerr << "nlag=" << nlag;
  cerr << "blockDim=" << blockDim.x << "," << blockDim.y << "," << blockDim.z << "," << endl;
  cerr << "gridDim="  << gridDim.x << "," << gridDim.y << "," << gridDim.z << "," << endl;
  
  unsigned lagbinplan_size = binplan_size * nlag;
  
  cycFoldIndPol <<<gridDim,blockDim,0,stream>>>(input,
                input_span,
                d_lagdata,
                lagbinplan_size,
                nlag,
                d_binplan);

  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::CyclicFoldEngineCUDA::fold cuda error: ");
}

