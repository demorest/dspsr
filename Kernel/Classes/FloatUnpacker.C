/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FloatUnpacker.h"

#include "Error.h"

#include <string.h>

// #define _DEBUG 1

using namespace std;

//! Null constructor
dsp::FloatUnpacker::FloatUnpacker (const char* _name)
  : Unpacker (_name)
{
  if (verbose)
    cerr << "dsp::FloatUnpacker ctor" << endl;
}

bool dsp::FloatUnpacker::matches (const Observation* observation)
{
  return
    observation->get_nbit() == 32 && 
    observation->get_machine() == "dspsr";
}

//! Return true if the unpacker support the specified output order
bool dsp::FloatUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::FloatUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

//! The unpacking routine
void dsp::FloatUnpacker::unpack ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const float* from_base = reinterpret_cast<const float*>(input->get_rawptr());
  
  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
  {
    const uint64_t stride = nchan * npol * ndim;

    for (unsigned ichan=0; ichan<nchan; ichan++) 
    {
      for (unsigned ipol=0; ipol<npol; ipol++) 
      {
	float* into = output->get_datptr (ichan, ipol);

	const float* from = from_base + ichan*npol*ndim + ipol*ndim;

	for (uint64_t idat=0; idat < ndat; idat++)
	{
	  for (unsigned idim=0; idim < ndim; idim++)
	    into[idim] = from[idim];

	  into += ndim;
	  from += stride;
	}
      }
    }

    break;
  }

  case TimeSeries::OrderTFP:
  {
    // the Dump operation outputs floats in TFP-major order

    float* into = output->get_dattfp();

    const uint64_t nfloat = ndat * nchan * npol * ndim;
    memcpy (into, from_base, nfloat * sizeof(float));

    break;
  }

  default:
    throw Error (InvalidState, "dsp::FloatUnpacker::unpack",
		 "unrecognized order");
  }
}
