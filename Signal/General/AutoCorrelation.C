/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/AutoCorrelation.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Scratch.h"

using namespace std;

dsp::AutoCorrelation::AutoCorrelation () 
  : Transformation<TimeSeries,TimeSeries> ("AutoCorrelation", anyplace, true)
{
  nlag = 1;
}

void dsp::AutoCorrelation::transformation ()
{
  if (verbose)
    cerr << "dsp::AutoCorrelation::transformation input ndat=" 
	 << get_input()->get_ndat() << endl;

  uint64   ndat  = input->get_ndat();
  unsigned npol  = input->get_npol();
  unsigned nchan = input->get_nchan();
  unsigned ndim  = input->get_ndim();

  if (ndim != 1)
    throw Error (InvalidState, "dsp::AutoCorrelation::transformation",
                 "can handle only ndim == 1 data");

  unsigned nlaghalf = nlag / 2;
  unsigned required = 2 * nlag;

  if (verbose)
    cerr << "Convolution::transformation nlag=" << nlag 
	 << " nlaghalf=" << nlaghalf << endl;

  // there must be at least enough data for one FFT
  if (ndat < required)
    throw Error (InvalidState, "dsp::Convolution::transformation",
		 "error ndat="I64" < min=%d", ndat, required);

  // number of FFTs for this data block
  unsigned long npart = (ndat-nlag)/nlag;

  if (verbose)
    cerr << "Convolution::transformation npart=" << npart << endl;

  // prepare the output TimeSeries
  output->copy_configuration (input);
  get_output()->set_nchan( nchan );
  get_output()->set_npol( npol );
  get_output()->set_ndim( nlag );
  get_output()->set_domain( "Lag" );

  get_output()->set_rate (get_input()->get_rate()/nlag);

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());
  if (weighted_output) {
    weighted_output->convolve_weights (nlag*2, nlag);
    weighted_output->scrunch_weights (nlag);
  }

  if (input.get() == output.get())
    output->set_ndat (npart);
  else
    output->resize (npart);

  // get_output()->check_sanity();
  //  double scale = input->get_scale ();

  // cerr << "scale=" << scale << endl;

  // nfilt_pos complex points are dropped from the start of the first FFT
  output->change_start_time (nlag);
  output->rescale (nlag);

  // temporary things that should not go in and out of scope
  const float* iptr = 0;
  const float* jptr = 0;
  float* outptr = 0;
  unsigned ilag, jlag;
  float total;
  float product;

  float* copy = scratch->space<float> (nlag);
  const unsigned nbytes = nlag * sizeof(float);

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned ipol=0; ipol < npol; ipol++) {

      // rescale the data
      //outptr = const_cast<float*>(input->get_datptr (ichan, ipol));
      //for (unsigned idat=0; idat < ndat; idat++)
      //outptr[idat] /= scale;

      for (unsigned ipart=0; ipart < npart; ipart++)  {
	
	uint64 offset = ipart * nlag;
	
	iptr = input->get_datptr (ichan, ipol) + offset;

	memcpy (copy, iptr + nlaghalf, nbytes);
	jptr = copy;

        outptr = output->get_datptr (ichan, ipol) + offset;

	for (ilag=0; ilag < nlag; ilag++) {
	  total = 0;
	  for (jlag=0; jlag < nlag; jlag++) {
	    product = iptr[jlag] * jptr[jlag];

#ifdef _DEBUG	    
	    if (!finite(product))
	      throw Error (InvalidParam,
			   "dsp::AutoCorrelation::transformation",
			   "product not finite."
			   " ichan=%d ipol=%d ipart=%d ilag=%d jlag=%d",
			   ichan, ipol, ipart, ilag, jlag);
#endif

	    total += product;
	  }

#ifdef _DEBUG	    
	  if (!finite(total))
	    throw Error (InvalidParam, "dsp::AutoCorrelation::transformation",
			 "total not finite. ichan=%d ipol=%d ipart=%d ilag=%d",
			 ichan, ipol, ipart, ilag);
#endif

	  outptr[ilag] = total;
	  iptr ++;
	}

      }  // for each part of the time series

    } // for each poln

  // for each channel

}

