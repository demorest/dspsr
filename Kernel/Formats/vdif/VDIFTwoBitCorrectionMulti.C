/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitCorrectionMulti.h"
#include "dsp/TwoBitTable.h"

bool dsp::VDIFTwoBitCorrectionMulti::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" 
      && observation->get_nbit() == 2
      && observation->get_npol() == 2;
}

//! Null constructor
dsp::VDIFTwoBitCorrectionMulti::VDIFTwoBitCorrectionMulti ()
  : SubByteTwoBitCorrection ("VDIFTwoBitCorrectionMulti")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  set_ndig(4);
}
/*! Each 2-bit sample from each digitizer is packed into one byte */
unsigned dsp::VDIFTwoBitCorrectionMulti::get_ndig_per_byte () const
{
  if (input->get_state() == Signal::Analytic)
  	return 4;
  else 
	return 2;
}
unsigned dsp::VDIFTwoBitCorrectionMulti::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The in-phase and quadrature components must be interleaved */
unsigned dsp::VDIFTwoBitCorrectionMulti::get_output_incr () const
{
  return 2;
}
/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::VDIFTwoBitCorrectionMulti::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::VDIFTwoBitCorrectionMulti::get_shift (unsigned idig, unsigned isamp) const
{
  
    
  if (input->get_state() == Signal::Analytic) {
    unsigned shift[4] = {4,6,0,2};

    return shift[idig] ;
  }
  else {
   // this was the original default shift
    return (idig + isamp * 2) * 2; 
  }
  
}

