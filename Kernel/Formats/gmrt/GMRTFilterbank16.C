/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTFilterbank16.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::GMRTFilterbank16::GMRTFilterbank16 (const char* name)
  : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::GMRTFilterbank16::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::GMRTFilterbank16::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::GMRTFilterbank16::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::GMRTFilterbank16::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  return observation->get_machine() == "PA" 
    && observation->get_nbit() == 16
    && observation->get_ndim() == 1
    && observation->get_npol() == 1;
}

unsigned dsp::GMRTFilterbank16::get_output_ipol (unsigned idig) const
{
  return 0;
}

unsigned dsp::GMRTFilterbank16::get_output_ichan (unsigned idig) const
{
  return idig;
}

void dsp::GMRTFilterbank16::unpack ()
{
  const uint64 ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();

  const uint16* base = reinterpret_cast<const uint16*>(input->get_rawptr());

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
    {
      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        const uint16* from = base + ichan;
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
      const uint16* from = base;
      float* into = output->get_dattfp();

      const uint64 nfloat = nchan * ndat;
      for (uint64 ifloat=0; ifloat < nfloat; ifloat ++)
	into[ifloat] = float( from[ifloat] );
    }
    break;

  default:
    throw Error (InvalidState, "dsp::GMRTFilterbank16::unpack",
		 "unrecognized order");
    break;
  }
}

