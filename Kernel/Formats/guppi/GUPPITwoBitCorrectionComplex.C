/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPITwoBitCorrectionComplex.h"
#include "dsp/TwoBitTable.h"

bool dsp::GUPPITwoBitCorrectionComplex::matches (const Observation* observation)
{
  return observation->get_machine().substr(1) == "UPPI" 
      && observation->get_nbit() == 2
      && observation->get_npol() == 2
      && observation->get_state() == Signal::Analytic;
}

//! Null constructor
dsp::GUPPITwoBitCorrectionComplex::GUPPITwoBitCorrectionComplex ()
  : SubByteTwoBitCorrection ("GUPPITwoBitCorrectionComplex")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  //table->set_reverse_2bit();
  //table->rebuild();
}

// Note this could be uncommented to consider the real and imaginary parts
// to come from a single "digitizer". Not totally clear what the
// consequences are. 
//unsigned dsp::GUPPITwoBitCorrectionComplex::get_ndim_per_digitizer () const
//{
//  return 2;
//}

// The following assume we are using ndim_per_digitizer==1
// This means there are 4 digitizers per byte

unsigned dsp::GUPPITwoBitCorrectionComplex::get_ndig_per_byte () const
{
  return 4;
}

unsigned dsp::GUPPITwoBitCorrectionComplex::get_input_offset (unsigned idig) 
  const
{
  return (idig/4);
}

unsigned dsp::GUPPITwoBitCorrectionComplex::get_input_incr () const
{
  return input->get_nchan();
}

unsigned dsp::GUPPITwoBitCorrectionComplex::get_output_ichan (unsigned idig) 
  const
{
  return (idig/4);
}

unsigned dsp::GUPPITwoBitCorrectionComplex::get_output_ipol (unsigned idig) 
  const
{
  return (idig/2) % 2;
}

unsigned
dsp::GUPPITwoBitCorrectionComplex::get_shift (unsigned idig, unsigned isamp) 
  const
{
  return (idig%4) * 2;
}
