/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcOne.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::SigProcOne::SigProcOne (const char* name) : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::SigProcOne::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::SigProcOne::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::SigProcOne::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::SigProcOne::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  return observation->get_machine() == "SigProc" 
    && observation->get_nbit() == 1
    && observation->get_ndim() == 1
    && observation->get_npol() == 1;
}

unsigned dsp::SigProcOne::get_output_ipol (unsigned idig) const
{
  return 0;
}

unsigned dsp::SigProcOne::get_output_ichan (unsigned idig) const
{
  return idig;
}

void dsp::SigProcOne::unpack ()
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
        const unsigned char* from = input->get_rawptr() + ichan/8;
	const unsigned shift = (ichan % 8);
        const unsigned nchan8 = nchan/8;

	float* into = output->get_datptr (ichan, 0);

	for (unsigned bt = 0; bt < ndat; bt++)
        {
          // shift and mask one channel
          into[bt] = float( ((*from) >> shift) & 0x01 );
          // skip to next byte in which
	  from += nchan8;
	}
      }
    }
    break;

  case TimeSeries::OrderTFP:
    {
      const unsigned char* from = input->get_rawptr();
      float* into = output->get_dattfp();

      const uint64 nfloat = nchan * ndat;
      for (uint64 ifloat=0; ifloat < nfloat; ifloat += 8)
      {
	unsigned char byte = *from;

	*into = float( byte & 0x01 ); into ++;
	*into = float( (byte >> 1) & 0x01 ); into ++;
	*into = float( (byte >> 2) & 0x01 ); into ++;
	*into = float( (byte >> 3) & 0x01 ); into ++;
	*into = float( (byte >> 4) & 0x01 ); into ++;
	*into = float( (byte >> 5) & 0x01 ); into ++;
	*into = float( (byte >> 6) & 0x01 ); into ++;
	*into = float( (byte >> 7) & 0x01 ); into ++;

      }
    }
    break;

  default:
    throw Error (InvalidState, "dsp::SigProcOne::unpack",
		 "unrecognized order");
    break;
  }
}

