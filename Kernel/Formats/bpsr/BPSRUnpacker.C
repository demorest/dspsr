/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BPSRUnpacker.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::BPSRUnpacker::BPSRUnpacker (const char* name) : HistUnpacker (name)
{
}

//! Return true if the unpacker support the specified output order
bool dsp::BPSRUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return true;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::BPSRUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::BPSRUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::BPSRUnpacker::matches \n"
      " machine=" << observation->get_machine() << " should be BPSR \n"
      " state=" << observation->get_state() << " should be PPQQ \n"
      " npol=" << observation->get_npol() << " should be 2 \n"
      " nbit=" << observation->get_nbit() << " should be 8 \n"
      " ndim=" << observation->get_ndim() << " should be 1 \n"
         << endl;

  return observation->get_machine() == "BPSR" 
    && observation->get_state() == Signal::PPQQ
    && observation->get_npol() == 2
    && observation->get_nbit() == 8
    && observation->get_ndim() == 1;
}

/*! The first nchan digitizer channels are poln0, the next nchan are poln1 */
unsigned dsp::BPSRUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / input->get_nchan();
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::BPSRUnpacker::get_output_ichan (unsigned idig) const
{
  return idig % input->get_nchan();
}

void dsp::BPSRUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
    {
      const unsigned step = npol * nchan;

      // input data are organized: p0c0 p0c1 p1c0 p1c1 p0c2 p0c3 p1c2 p1c3 ...

      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        unsigned chan_off = (ichan/2) * 4 + ichan%2;

        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          unsigned pol_off = ipol * 2;

          const unsigned char* from = input->get_rawptr() + chan_off + pol_off;
          float* into = output->get_datptr (ichan, ipol);

          // unsigned long* hist = get_histogram (off);

          for (unsigned bt = 0; bt < ndat; bt++)
          {
            // hist[ *from ] ++;
            into[bt] = float( *from );
            from += step;
          }
        }
      }
    }
break;
  case TimeSeries::OrderTFP:
    {
      const unsigned char* from = input->get_rawptr();
      float* into = output->get_dattfp();

      const uint64_t nfloat = npol * nchan * ndat;
      for (uint64_t ifloat=0; ifloat < nfloat; ifloat += 4)
      {
        into[0] = float( from[0] );
        into[1] = float( from[2] );
        into[2] = float( from[1] );
        into[3] = float( from[3] );
        
        into += 4;
        from += 4;
      }
    }
break;
  default:
    throw Error (InvalidState, "dsp::BPSRUnpacker::unpack",
                 "unrecognized order");
break;
  }
}

