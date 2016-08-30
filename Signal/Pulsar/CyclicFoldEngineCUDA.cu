//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012 by Glenn Jones and Paul Demorest
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
  use_set_bins = true;


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
    cerr << "CUDA::CyclicFoldEngineCUDA::synch this=" << this << " synchronised=" << synchronized <<  endl;

  if (synchronized)
    return;

  if (dsp::Operation::verbose) {
	  cerr << "CUDA::CyclicFoldEngineCUDA::synch output=" << output << endl;

	  cerr << "CUDA::CyclicFoldEngineCUDA:: transferring lag data synch out=" << out <<" out.ndat_folded=" << out->get_ndat_folded()
			  << " lagdata_size=" << lagdata_size
		 <<endl; // << " output.ndatfolded" << output->get_ndat_folded() << endl;
  }
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

  cudaThreadSynchronize();
  /*
  cerr << "synch stream=" <<stream <<endl;
  for (int i = 0; i < nlag*nchan*2*2*8; i++){
	  cerr << lagdata[i] << " ";
  }
  cerr << endl << endl;
  for(int ibin = 0; ibin < nbin; ibin++){
	  for(int ipol = 0; ipol < npol; ipol++){
		  cerr << "\nibin=" << ibin << " ipol=" << ipol;
		  for(int ichan = 0; ichan < nchan; ichan++){
			  for(int ilag = 0; ilag < nlag; ilag++){
				  float x = lagdata[2*(ibin*npol*nchan*nlag + ipol*nchan*nlag + ichan*nlag + ilag)];
				  float y = lagdata[2*(ibin*npol*nchan*nlag + ipol*nchan*nlag + ichan*nlag + ilag) + 1];
				  cerr <<  " " << x << "," << y;

			  }
			  cerr << "\n|" << ichan << "\n";
		  }
	  }
  }
  */
  // Call usual synch() to do transform
  dsp::CyclicFoldEngine::synch(out);

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch now synch'ed" << endl;

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
/*// FOllowing moved to set_bins
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
  */
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
    memset(lagdata, 0, sizeof(float)*lagdata_size); // this could be removed once we're convinced thigs are working since the cuda memory is zeroed as well
    
    if (d_lagdata) cudaFree(d_lagdata);
    cudaMalloc((void**)&d_lagdata, lagdata_size * sizeof(float));
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
    
  }


}

void CUDA::CyclicFoldEngineCUDA::set_bin (uint64_t idat, double d_ibin, 
        double bins_per_sample)
{
	return;
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

uint64_t CUDA::CyclicFoldEngineCUDA::get_bin_hits (int ibin)
{
	int iturn = 0;
	int idx = 0;
	idx = iturn*nbin*nlag + ibin*nlag; // we want the zero lag hits
	uint64_t hits = 0;
//	cerr << "ibin: " << ibin << " ";
	while (idx < binplan_size) {
		hits += lagbinplan[idx].hits;
		//cerr << lagbinplan[idx].hits << " ";
		iturn += 1;
		idx = iturn*nbin*nlag + ibin*nlag; // we want the zero lag hits
	}
	if(parent->verbose){
		cerr << "CyclicFoldEngineCUDA::get_bin_hits ibin=" << ibin << " hits=" << hits << endl;
	}
	return hits;
}
  
// set_bins was added as a more efficient way of setting up the bin plan all in one go, rather than through repeated redundant calculations
// as was previously done using set_bin
// The bin plan is indexed as iturn*nbin*nlag + ibin*nlag + ilag
// each entry indicates the starting data sample (offset), the number of data samples to include in this lag/bin (hits), and the bin index (ibin)
// there is one entry for every lag, every bin, and for all turns in this data block.
uint64_t CUDA::CyclicFoldEngineCUDA::set_bins (double phi, double phase_per_sample, uint64_t _ndat, uint64_t idat_start)
{
	if(parent->verbose){
		cerr << "Got to CUDA::CyclicFoldEngineCUDA::set_bins" << endl;
	}


	phi = phi - floor(phi); // fractional phase at start
	double samples_per_bin = (1.0 / nbin) * (1.0 / phase_per_sample); // (1 turn / nbin bins) * (turns (phase) / sample) ^ -1
	double nturns = _ndat * phase_per_sample; // total number of turns represented by this block of data
	double minph,maxph;
	double startph = phi;  //starting fractional phase, the smallest valid phase
	double endph = startph + nturns; // final phase, the largest valid phase of any data point
	int startdat = 0;
	int intnturns = ceil(nturns) + 1;  // total number of turns in the binplan. This could probably be safely just ceil(nturns) but add 1 to be sure.
	int iturn,ibin,ilag;
	int planidx;

	int _binplan_size = intnturns*nbin*nlag; // total number of entries in the bin plan.

	if(parent->verbose) {
		cerr << "Start ph:" << startph << " intnturns:" <<intnturns << " _ndat:" << _ndat << " nlag:" << nlag
				<< " phase per sample:" << phase_per_sample<< " nturns:" << nturns << endl ;
		cerr << "binplansize:" << binplan_size << "  _binplansize:" << _binplan_size << endl;
	}
	// allocate memory for the binplan
	  if (_binplan_size > binplan_size) {

		    if (parent->verbose)
		      cerr << "dsp::CyclicFoldEngine::set_bins alloc binplan" << endl;

		    if (lagbinplan) {
		      delete [] lagbinplan;
		    }
		    lagbinplan = new bin [_binplan_size];


		    binplan_size = _binplan_size;
		  }
	  memset(lagbinplan, 0 , sizeof(bin)*_binplan_size);  // all entries start out with zero hits, so any uninitialized portions will be ignored by the folding threads
	  ndat_fold = _ndat;

	for (iturn=0;iturn < intnturns; iturn++){
		for (ibin = 0; ibin < nbin; ibin++) {
			for (ilag=0; ilag < nlag; ilag++) {
				// minph is the starting phase of valid data for this lag/bin
				// maxph is the ending phase
				// thus we want to include all data points with phases in between minph and maxph
				minph = (ibin*1.0)/nbin + iturn + (ilag*phase_per_sample)/2.0;
				maxph = (ibin+1.0)/nbin + iturn + (ilag*phase_per_sample)/2.0;
				// index of this binplan entry
				planidx = iturn*nbin*nlag + ibin*nlag + ilag;

				if ( maxph > endph ) {
					maxph = endph; // keep maxph from going off the end of the data block. In theory we should really pull more data from the next block, but for now
									// we just ignore correlations that span more than one data block
				}
				if ((minph > endph) || (maxph < minph)) {
					// if the start of this lag/bin data is past the end of the data block (minph > endph), there is no valid data for this lag/bin
					// if maxph < minph, then it must be that minph > endph because the only way for this to happen would be if maxph were reassigned to endph in the previous clause.
					lagbinplan[planidx].offset = 0;
					lagbinplan[planidx].ibin = 0;
					lagbinplan[planidx].hits = 0;
					continue;
				}

				if (minph > startph){
					// The basic case, the lag/bin data is fully within the data block, or goes right up to the end of the block (in which case maxph=endph)
					lagbinplan[planidx].offset = round((minph-startph)/phase_per_sample);
					lagbinplan[planidx].ibin = ibin;
					lagbinplan[planidx].hits = round((maxph-minph)/phase_per_sample);
				}
				else if (maxph > startph){
					// In this case, the start of the lag/bin data precedes the first available data point, but there is still valid data from startph to maxph
//					cerr << "minph < startph " << minph << " < " << startph << endl;
					lagbinplan[planidx].offset = 0;
					lagbinplan[planidx].ibin = ibin;
					lagbinplan[planidx].hits = round((maxph-startph)/phase_per_sample);
				}
				else {
					// Finally, here minph <= startph and maxph <= startph, so the data needed fully precedes this data block.
//					cerr << "maxph < startph " << minph << " < " << startph << endl;
					lagbinplan[planidx].offset = 0;
					lagbinplan[planidx].ibin = 0;
					lagbinplan[planidx].hits = 0;
				}
/*				if (ilag == 4) {
				cerr << "iturn,ibin,ilag: " << iturn << "," << ibin << "," << ilag << ","
						<< " offset=" << lagbinplan[planidx].offset
						<< " hits=" << lagbinplan[planidx].hits
						<< " minph=" << minph
						<< " maxph=" << maxph

						<< endl;
				} */
			}
		}
	}
	return ndat_fold;
}


void CUDA::CyclicFoldEngineCUDA::zero ()
{
  dsp::CyclicFoldEngine::zero();
  if (d_lagdata && lagdata_size>0) {
	  if(parent->verbose)
		  cerr << "CUDA::CyclicFoldEngineCUDA::zero: zeroing lagdata on gpu" << endl;
    if (stream)
      cudaMemsetAsync(d_lagdata, 0, lagdata_size * sizeof(float), stream);
    else
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
  }
  else {
	  if(parent->verbose)
		  cerr << "CUDA::CyclicFoldEngineCUDA::zero: not doing anything because d_lagdata=" <<d_lagdata << " and lagdata_size=" << lagdata_size << endl;
  }
}

void CUDA::CyclicFoldEngineCUDA::send_binplan ()
{

  uint64_t mem_size = binplan_size * sizeof(bin);

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan ndat=" << ndat_fold
         << "  Allocating on device mem_size " << mem_size
         << " binplan_size=" << binplan_size
         << " nlag=" << nlag
         << " sizeof(bin)=" << sizeof(bin)
         << " current_turn=" << current_turn
//         << " orig_size=" << orig_size
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
	  //cerr << "orig_size=" << orig_size << "< binplansize=" << binplan_size << "so freeing.." << endl;
	  error =cudaFree(d_binplan);
	  if (error != cudaSuccess)
		  throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
					   "cudaFree %s %s",
					   stream?"Async":"", cudaGetErrorString (error));
	  if(parent->verbose)
		  cerr << "realocating..." << endl;
	  error = cudaMalloc ((void **)&(d_binplan),mem_size);
	  if (error != cudaSuccess)
		  throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
					   "cudaMalloc new %s %s",
					   stream?"Async":"", cudaGetErrorString (error));
  }


/*  ofstream fbin;
  fbin.open("cudabinplan.dat", ios::binary | ios::app);
  fbin.write((char *)(lagbinplan),mem_size);
  cerr << "done, dumping cudabinplan, closing files" << endl;
  fbin.close();
*/
  if(parent->verbose){
  cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan copying: stream=" << stream << " d_binplan=" << d_binplan << " mem_size=" << mem_size <<
		  " lagbinplan=" << lagbinplan << endl;
  }
  if (stream)
    error = cudaMemcpyAsync (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice,stream);
  else
    error = cudaMemcpy (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
                 "cudaMemcpy%s %s",
                 stream?"Async":"", cudaGetErrorString (error));
}

// This function is never used. Lagdata is trasfered by the synch call
void CUDA::CyclicFoldEngineCUDA::get_lagdata ()
{
	if(parent->verbose)
		cerr << "getting lagdata" << endl;
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
// Since there is a maximum number of threads per block which may be less than the number of lags times number of pols,
// the ilag index is split into ilag = ilagb*nlaga + ilaga, where nlaga will be such that nlaga*npol = max_threads_per_block
// Each thread calculates the cyclic correlation for one lag for one bin for one input channel for one pol
// threadIdx.x -> ilaga    blockDim.x
// threadIdx.y -> pol
// threadIdx.z -> not used
// blockIdx.x -> ilagb
// blockIdx.y -> ibin
// blockIdx.z = ichan

// data is in FPT order, so chunks of time for a given pol and frequency
// in_span gives size of one time chunk for a given freq and pol in floats
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

  in_base  += in_span  * (ichan*npol + ipol);	//in_span is in units of float, so no need to mult by ndim
//  out_base += out_span * (ichan*npol + ipol);
  out_base += ndim*(ibin*npol*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag
    + ilag);

  unsigned bpstep = nlag*nbin; // step size to get to the next rotation for a given lag and bin in the binplan

/*  for(int a=20; a < 256*64*33*2; a++){
	  out_base[2*a] = float(a);
  }
  out_base[0] = float(nchan);
  out_base[2] = float(npol);
  out_base[4] = float(nbin);
  out_base[6] = float(nlag);
  out_base[8] = float(ndim);
  out_base[10] = float(npol*nchan*nlag*ndim);
  out_base[12] = float(nchan*nlag*ndim);
  out_base[14] = float(nlag*ndim);
  out_base[16] = float((nchan*nlag + nlag + npol*nchan*nlag)*ndim);
  out_base[18] = float(nchan);
  out_base[20] = float(nchan*nlag*ndim);
  out_base[22] = 9.87612;


  out_base[ndim*(ibin*npol*nchan*nlag
			+ ipol*nchan*nlag
			+ ichan*nlag
			+ ilag) + 1] = (ipol<<8) + ilag;
*/
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

// Since there is a maximum number of threads per block which may be less than the number of lags times number of pols,
// the ilag index is split into ilag = ilagb*nlaga + ilaga, where nlaga will be such that nlaga*npol = max_threads_per_block
// Each thread calculates the cyclic correlation for one lag for one bin for one input channel for one pol
// threadIdx.x -> ilaga    blockDim.x
// threadIdx.y -> pol
// blockIdx.x -> ilagb
// blockIdx.y -> ibin
// This version gets passed ichan and nchan directly (it operates just on one channel) because early cuda could not handle 3dim thread grids
// data is in FPT order, so chunks of time for a given pol and frequency
// in_span gives size of one time chunk for a given freq and pol in floats
__global__ void cycFoldIndPolOneChan (const float* in_base,
                unsigned in_span,
                float* out_base,
                unsigned binplan_size,
                unsigned nlag,
                CUDA::bin* binplan,
                unsigned nchan,
                unsigned ichan)
{
  unsigned ilaga = threadIdx.x;
  unsigned nlaga = blockDim.x;
  unsigned ilagb = blockIdx.x;
  unsigned ibin = blockIdx.y;
  unsigned ipol = threadIdx.y;
  unsigned npol = blockDim.y;
  unsigned nbin = gridDim.y;
  unsigned ilag = ilagb*nlaga + ilaga;
  if (ilag >= nlag){
	  return;
  }
  unsigned planidx = nlag*ibin+ilag;
  const unsigned ndim = 2; // always complex data assumed

  if (planidx >= binplan_size) {
    return;
  }

  in_base  += in_span  * (ichan*npol + ipol);	//in_span is in units of float, so no need to mult by ndim
//  out_base += out_span * (ichan*npol + ipol);
  out_base += ndim*(ibin*npol*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag
    + ilag);

  unsigned bpstep = nlag*nbin; // step size to get to the next rotation for a given lag and bin in the binplan

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
	if(parent->verbose)
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
//  dim3 gridDim (nlagb, nbin, nchan);
  dim3 gridDim (nlagb, nbin, 1);
  if(parent->verbose){
	  cerr << "nlag=" << nlag << " binplan_size=" << binplan_size << " input_span=" << input_span  << " d_lagdata=" << d_lagdata << endl;
	  cerr << "blockDim=" << blockDim.x << "," << blockDim.y << "," << blockDim.z << "," << endl;
	  cerr << "gridDim="  << gridDim.x << "," << gridDim.y << "," << gridDim.z << "," << endl;
  }
  unsigned lagbinplan_size = binplan_size;

  for(unsigned ichan=0;ichan < nchan; ichan++){
	  cycFoldIndPolOneChan <<<gridDim,blockDim,0,stream>>>(input,
					input_span,
					d_lagdata,
					lagbinplan_size,
					nlag,
					d_binplan,
					nchan,
					ichan);
  }
  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;
  cudaThreadSynchronize();

  if(parent->verbose)
	  cerr << "CyclicFoldEngineCUDA::fold finished, syncronized=false" << endl;

  //if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::CyclicFoldEngineCUDA::fold cuda error: ");
}

