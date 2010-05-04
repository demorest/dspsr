/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcUnpacker.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::SigProcUnpacker::SigProcUnpacker (const char* name) : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::SigProcUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::SigProcUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::SigProcUnpacker::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::SigProcUnpacker::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  unsigned nbit = observation->get_nbit();

  return observation->get_machine() == "SigProc" 
    && observation->get_ndim() == 1
    && observation->get_npol() == 1
    && ( nbit==1 || nbit==2 || nbit==4 || nbit==8 );
}

unsigned dsp::SigProcUnpacker::get_output_ipol (unsigned idig) const
{
  return 0;
}

unsigned dsp::SigProcUnpacker::get_output_ichan (unsigned idig) const
{
  return idig;
}

void dsp::SigProcUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned nbit = input->get_nbit();

  // these tricks do quick division and modulus
  unsigned div_shift = 0;
  unsigned mod_mask = 0;
  unsigned char keep_mask = 0xff;

  switch ( nbit )
  {
  case 1:
    div_shift = 3;   // divide by 8
    mod_mask = 0x07;
    keep_mask = 0x01;
    break;
  case 2:
    div_shift = 2;   // divide by 4
    mod_mask = 0x03;
    keep_mask = 0x03;
    break;
  case 4:
    div_shift = 1;   // divide by 2
    mod_mask = 0x01;
    keep_mask = 0x0f;
    break;
  }

  // e.g. number of bytes spanned by each spectrum = nchan * nbit / 8
  const unsigned nchan_bytes = nchan >> div_shift;

  const unsigned char* from_base = input->get_rawptr();
  
  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
  {
    for (unsigned ichan=0; ichan<nchan; ichan++) 
    {
      // from = from_base + ichan * nbit / 8
      const unsigned char* from = from_base + (ichan >> div_shift);

      // shift = ichan % (8/nbit)
      const unsigned shift = (ichan & mod_mask);
 
      float* into = output->get_datptr (ichan, 0);
      
      if (nbit == 8)
	for (unsigned bt = 0; bt < ndat; bt++)
        {
          into[bt] = float( *from );
	  from += nchan;
	}
      else
	for (unsigned bt = 0; bt < ndat; bt++)
	{
	  // shift and mask one channel
	  into[bt] = float( ((*from) >> shift) & keep_mask );
	  // skip to next byte in which this channel occurs
	  from += nchan_bytes;
	}
    }

    break;
  }

  case TimeSeries::OrderTFP:
  {
    float* into = output->get_dattfp();

    const uint64_t nbyte = nchan_bytes * ndat;
    const unsigned samples_per_byte = 8/nbit;

    if (nbit == 8)
      for (uint64_t ibyte=0; ibyte < nbyte; ibyte ++)
	into[ibyte] = float( from_base[ibyte] );
    else
      for (uint64_t ibyte=0; ibyte < nbyte; ibyte++)
      {
	unsigned char byte = from_base[ibyte];
	
	for (unsigned i=0; i<samples_per_byte; i++)
	{
	  *into = float( (byte >> i*nbit) & keep_mask );
	  into ++;
	}
      }

    break;
  }

  default:
    throw Error (InvalidState, "dsp::SigProcUnpacker::unpack",
		 "unrecognized order");
  }
}

