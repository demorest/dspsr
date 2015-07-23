/***************************************************************************
 *
 *   Copyright (C) 2008-2014 by Andrew Jameson & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BPSRCrossUnpacker.h"
//#include "dsp/DADABuffer.h"
#include "dsp/ASCIIObservation.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::BPSRCrossUnpacker::BPSRCrossUnpacker (const char* name) : HistUnpacker (name)
{
  gain_polx = -1;
  unpack_ppqq_only = false;
}

//! Return true if the unpacker support the specified output order
bool dsp::BPSRCrossUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return true;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::BPSRCrossUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::BPSRCrossUnpacker::set_output_ppqq ()
{
  //cerr << "dsp::BPSRCrossUnpacker::set_output_ppqq" << endl;
  unpack_ppqq_only = true;
  output->set_npol(2);
  output->set_state(Signal::PPQQ);
}

bool dsp::BPSRCrossUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::BPSRCrossUnpacker::matches \n"
      " machine=" << observation->get_machine() << " should be BPSR \n"
      " state=" << observation->get_state() << " should be Coherence\n"
      " npol=" << observation->get_npol() << " should be 4 \n"
      " nbit=" << observation->get_nbit() << " should be 8 \n"
      " ndim=" << observation->get_ndim() << " should be 1 \n"
         << endl;

  return observation->get_machine() == "BPSR" 
    && observation->get_state() == Signal::Coherence
    && observation->get_npol() == 4
    && observation->get_nbit() == 8
    && observation->get_ndim() == 1;
}

/*! The first nchan digitizer channels are poln0, the next nchan are poln1 */
unsigned dsp::BPSRCrossUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / input->get_nchan();
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::BPSRCrossUnpacker::get_output_ichan (unsigned idig) const
{
  return idig % input->get_nchan();
}

void dsp::BPSRCrossUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();

  const unsigned pol_offset_even[4] = {0, 2, 4, 5};
  const unsigned pol_offset_odd[4]  = {1, 3, 6, 7};

  if (unpack_ppqq_only)
  {
    //cerr << "dsp::BPSRCrossUnpacker::unpack: npol=2 state=PPQQ" << endl;
    output->set_npol(2);
    output->set_state(Signal::PPQQ);
    output->resize(ndat);
  }

  if (gain_polx < 0)
  {
    const Input * in = input->get_loader();
    const Observation * obs = in->get_info();
    const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
    if (info)
    {
      if (info)
      {
        // attempt to get the FACTOR_POLX from the header. This completely describes
        // the factor necessary to correct the AB* values
        try
        {
          if (info->custom_header_get ("FACTOR_POLX", "%f", &gain_polx) == 1)
          {
            if (verbose)
              cerr << "dsp::BPSRCrossUnpacker::unpack FACTOR_POLX=" << gain_polx << endl;
          }
        }
        catch (Error& error)
        {
          // older method that makes the assumption that the AA and BB are in
          // bit window 1. AB* is in bit window 3. The correct calculation is
          //    gain_polx = polx * 2^11 / (2^8 * (bwx - bw))
          unsigned polx;
          if (info->custom_header_get ("GAIN_POLX", "%u", &polx) == 1)
          {
            if (polx == 0)
            {
              gain_polx = 1;
            }
            else
            {
              gain_polx = ((float) polx) / 32;
            }
          }
          if (verbose)
            cerr << "dsp::BPSRCrossUnpacker::unpack GAIN_POLX=" << polx << " FACTOR_POLX=" << gain_polx << endl;
        }
      }
    }
  }

  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
    {
      if (verbose)
        cerr << "dsp::BPSRCrossUnpacker::unpack Output order OrderFPT" << endl;
    
      // input data are organized: 
      //   pAc0 pAc1 pBc0 pBc1 pABc0Re pABc0Im pABc1Re pABc1Im
      //   pAc2 pAc3 pBc2 pBc3 pABc2Re pABc2Im pABc3Re pABc3Im

      const unsigned step = npol * nchan;

      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        unsigned chan_off = (ichan/2) * 8;

        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          unsigned pol_off;
          if (ichan % 2)
            pol_off = pol_offset_odd[ipol];
          else
            pol_off = pol_offset_even[ipol];

          const unsigned char* from = input->get_rawptr() + chan_off + pol_off;
          float* into = output->get_datptr (ichan, ipol);
        
          if (ipol < 2)
          {
            // unsigned long* hist = get_histogram (off);
            for (unsigned bt = 0; bt < ndat; bt++)
            {
              // hist[ *from ] ++;
              into[bt] = float( *from );
              from += step;
            }
          }
          else
          {
            for (unsigned bt = 0; bt < ndat; bt++)
            {
              if (!unpack_ppqq_only)
                into[bt] = float( ((char) *from) ) / gain_polx;
              from += step;
            }
          }
        }
      }
    }
break;
  case TimeSeries::OrderTFP:
    {
      if (verbose)
        cerr << "dsp::BPSRCrossUnpacker::unpack Output order OrderTFP\n" << endl;

      const unsigned char* from = input->get_rawptr();
      float* into = output->get_dattfp();

      const uint64_t nfloat = npol * nchan * ndat;

      if (unpack_ppqq_only)
      {
        for (uint64_t ifloat=0; ifloat < nfloat; ifloat += 8)
        {
          into[0] = float( from[0] ) + 0.5;
          into[1] = float( from[2] ) + 0.5;
          into[2] = float( from[1] ) + 0.5;
          into[3] = float( from[3] ) + 0.5;

          into += 4;
          from += 8;
        }
      }
      else
      {
        for (uint64_t ifloat=0; ifloat < nfloat; ifloat += 8)
        {
          into[0] = float( from[0] ) + 0.5;
          into[1] = float( from[2] ) + 0.5;
          into[2] = float( ((char) from[4]) ) + 0.5;
          into[3] = float( ((char) from[5]) ) + 0.5;
          into[4] = float( from[1] ) + 0.5;
          into[5] = float( from[3] ) + 0.5;
          into[6] = float( ((char) from[6]) ) + 0.5;
          into[7] = float( ((char) from[7]) ) + 0.5;

          into[2] /= gain_polx;
          into[3] /= gain_polx;
          into[6] /= gain_polx;
          into[7] /= gain_polx;
          
          into += 8;
          from += 8;
        }
      }
    }
break;
  default:
    throw Error (InvalidState, "dsp::BPSRCrossUnpacker::unpack",
                 "unrecognized order");
break;
  }
}

