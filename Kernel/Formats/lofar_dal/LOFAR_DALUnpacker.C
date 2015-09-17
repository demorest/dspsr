/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LOFAR_DALUnpacker.h"

#include "Error.h"

#include <string.h>

// #define _DEBUG 1

using namespace std;

//! Null constructor
dsp::LOFAR_DALUnpacker::LOFAR_DALUnpacker ()
  : Unpacker ("LOFAR_DALUnpacker")
{
  if (verbose)
    cerr << "dsp::LOFAR_DALUnpacker ctor" << endl;
}

bool dsp::LOFAR_DALUnpacker::matches (const Observation* observation)
{
  return
    observation->get_nbit() == 32 && 
    observation->get_machine() == "COBALT";
}

//! Return true if the unpacker support the specified output order
bool dsp::LOFAR_DALUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::LOFAR_DALUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

//! The unpacking routine
void dsp::LOFAR_DALUnpacker::unpack ()
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
    const uint64_t stride = nchan * ndat;

    for (unsigned ipol=0; ipol<npol; ipol++) 
    {
      for (unsigned idim=0; idim<ndim; idim++) 
      {
	for (unsigned ichan=0; ichan<nchan; ichan++) 
	{
	  float* into = output->get_datptr (ichan, ipol) + idim;

	  const float* from = from_base + ichan;

	  for (uint64_t idat=0; idat < ndat; idat++)
	    into[idat*ndim] = from[idat*nchan];
	}
	from_base += stride;
      }
    }

#if 0
    for (unsigned ipol=0; ipol<npol; ipol++) 
    {
      float* data = output->get_datptr (53, ipol);
      for (unsigned i=0; i<10; i++)
	cerr << "ipol=" << ipol << " ifloat=" << data[i] << endl;
    }
#endif

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
    throw Error (InvalidState, "dsp::LOFAR_DALUnpacker::unpack",
		 "unrecognized order");
  }
}
