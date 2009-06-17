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
  : Unpacker (name)
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
    && (observation->get_npol() == 1 || observation->get_npol() == 4);
}

void dsp::GMRTFilterbank16::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();

  const int16_t* base = reinterpret_cast<const int16_t*>(input->get_rawptr());

  /*
    GMRT filterbank stores: RR, RL, LL, LR
    TimeSeries stores:      LL, RR, RL, LR
  */
  const unsigned ipol_map [4] = { 1, 2, 0, 3};

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
  {
    for (unsigned ipol=0; ipol<npol; ipol++) 
    {
      unsigned timeseries_ipol = 0;
      if (npol > 1)
	timeseries_ipol = ipol_map[ipol];

      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        const int16_t* from = base + ichan*npol + ipol;
	float* into = output->get_datptr (ichan, timeseries_ipol);
	
	for (unsigned bt = 0; bt < ndat; bt++)
	{
          into[bt] = float( *from );
	  from += nchan * npol;
	}
      }
    }
  }
  break;

  case TimeSeries::OrderTFP:
  {
    const int16_t* from = base;
    float* into = output->get_dattfp();

    const uint64_t nfloat = nchan * ndat;
    for (uint64_t ifloat=0; ifloat < nfloat; ifloat ++)
      into[ifloat] = float( from[ifloat] );
  }
  break;

  default:
    throw Error (InvalidState, "dsp::GMRTFilterbank16::unpack",
		 "unrecognized order");
    break;
  }
}

