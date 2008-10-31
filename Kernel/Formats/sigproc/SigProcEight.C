/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcEight.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::SigProcEight::SigProcEight (const char* name) : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::SigProcEight::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::SigProcEight::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::SigProcEight::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::SigProcEight::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  return observation->get_machine() == "SigProc" 
    && observation->get_nbit() == 8
    && observation->get_ndim() == 1
    && observation->get_npol() == 1;
}

unsigned dsp::SigProcEight::get_output_ipol (unsigned idig) const
{
  return 0;
}

unsigned dsp::SigProcEight::get_output_ichan (unsigned idig) const
{
  return idig;
}

void dsp::SigProcEight::unpack ()
{
  const uint64 ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
    {
      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        const unsigned char* from = input->get_rawptr() + ichan;
	float* into = output->get_datptr (ichan, 0);

	for (unsigned bt = 0; bt < ndat; bt++)
        {
          into[bt] = float( *from );
	  from += nchan;
	}
      }
    }
    break;

  case TimeSeries::OrderTFP:
    {
      const unsigned char* from = input->get_rawptr();
      float* into = output->get_dattfp();

      const uint64 nfloat = nchan * ndat;
      for (uint64 ifloat=0; ifloat < nfloat; ifloat ++)
	into[ifloat] = float( from[ifloat] );
    }
    break;

  default:
    throw Error (InvalidState, "dsp::SigProcEight::unpack",
		 "unrecognized order");
    break;
  }
}

