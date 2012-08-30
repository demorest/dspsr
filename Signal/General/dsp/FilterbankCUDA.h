//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankCUDA.h,v $
   $Revision: 1.17 $
   $Date: 2011/10/07 11:10:14 $
   $Author: straten $ */

#ifndef __FilterbankCUDA_h
#define __FilterbankCUDA_h

#include "dsp/FilterbankEngine.h"
#include "dsp/filterbank_cuda.h"

namespace CUDA
{
  class elapsed
  {
  public:
    elapsed ();
    void wrt (cudaEvent_t before);

    double total;
    cudaEvent_t after;
  };

  //! Discrete convolution filterbank step implemented using CUDA streams
  class FilterbankEngine : public dsp::Filterbank::Engine,
			   protected filterbank_cuda
  {
    unsigned nstream;

  public:

    //! Default Constructor
    FilterbankEngine (cudaStream_t stream);
    ~FilterbankEngine ();

    void setup (dsp::Filterbank*);
    void perform (const float* in);
    void finish ();

    int max_threads_per_block; 
  };

}

#endif
