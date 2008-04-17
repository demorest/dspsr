/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRTwoBitCorrection.h"

#include "dsp/Input.h"
#include "dsp/Observation.h"
#include "dsp/TwoBitTable.h"

using namespace std;

bool dsp::APSRTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "APSR"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Analytic;
}

//! Null constructor
dsp::APSRTwoBitCorrection::APSRTwoBitCorrection ()
  : TwoBitCorrection ("APSRTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::TwosComplement);
}


/*! 
  The real and imaginary components of the complex polyphase
  filterbank outputs are decimated together
*/
unsigned dsp::APSRTwoBitCorrection::get_ndim_per_digitizer () const
{
  return 2;
}

/*! The data from each polarization are written in blocks */
unsigned dsp::APSRTwoBitCorrection::get_input_incr () const
{
  return 1;
}

/*! The data from each polarization are separated by half the packet length */
unsigned dsp::APSRTwoBitCorrection::get_input_offset (unsigned idig) const
{
  unsigned resolution = input->get_loader()->get_resolution();
  unsigned offset = idig * input->get_nbytes(resolution) / 2;

  if (verbose)
    cerr << "dsp::APSRTwoBitCorrection::get_input_offset resolution=" << resolution 
         << " idig=" << idig << " offset=" << offset << endl;

  return offset;
}

