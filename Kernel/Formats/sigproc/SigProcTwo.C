/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcTwo.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::SigProcTwo::SigProcTwo (const char* name) : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::SigProcTwo::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::SigProcTwo::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::SigProcTwo::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::SigProcTwo::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  return observation->get_machine() == "SigProc" 
    && observation->get_nbit() == 2
    && observation->get_ndim() == 1
    && observation->get_npol() == 1;
}

unsigned dsp::SigProcTwo::get_output_ipol (unsigned idig) const
{
  return 0;
}

unsigned dsp::SigProcTwo::get_output_ichan (unsigned idig) const
{
  return idig;
}

void dsp::SigProcTwo::unpack ()
{
  const uint64 ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
    {
      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        // four channels in each byte
        const unsigned char* from = input->get_rawptr() + ichan/4;
	const unsigned shift = (ichan % 4) * 2;
        const unsigned nchan4 = nchan/4;

	float* into = output->get_datptr (ichan, 0);

	for (unsigned bt = 0; bt < ndat; bt++)
        {
          // shift and mask one channel
          into[bt] = float( ((*from) >> shift) & 0x03 );
          // skip to next byte in which
	  from += nchan4;
	}
      }
    }
    break;

  case TimeSeries::OrderTFP:
    {
      const unsigned char* from = input->get_rawptr();
      float* into = output->get_dattfp();

      const uint64 nfloat = nchan * ndat;
      for (uint64 ifloat=0; ifloat < nfloat; ifloat += 4)
      {
	unsigned char byte = *from;

	*into = float( byte & 0x03 ); into ++;
	*into = float( (byte >> 2) & 0x03 ); into ++;
	*into = float( (byte >> 4) & 0x03 ); into ++;
	*into = float( (byte >> 6) & 0x03 ); into ++;
      }
    }
    break;

  default:
    throw Error (InvalidState, "dsp::SigProcTwo::unpack",
		 "unrecognized order");
    break;
  }
}

