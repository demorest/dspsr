/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/FadcTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

#include <assert.h>

bool dsp::FadcTwoBitCorrection::matches (const Observation* observation)
{
  return (observation->get_nbit() == 2 && observation->get_state() == Signal::Analytic && observation->get_machine() == "Fadc");
}

//! Null constructor
dsp::FadcTwoBitCorrection::FadcTwoBitCorrection ()
  : SubByteTwoBitCorrection ("FadcTwoBitCorrection")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
}

//! Specialize the unpacker to the Observation                                                                                        
void dsp::FadcTwoBitCorrection::match (const Observation* observation)                                                             
{                                                                                                                                     
  int npol = observation->get_npol();
  if (npol==2)  set_ndig (4);
  else set_ndig (2);
}                                   

/*1 polarization: ADC0 ADC1 ADC0 ADC1  in one byte, i.e. 2 samples from 2 ADCs in each byte*/
/*2 polarizations: ADC0 ADC1 ADC2 ADC3  in one byte, i.e. 1 sample from 4 ADCs in each byte*/
unsigned dsp::FadcTwoBitCorrection::get_ndig_per_byte () const
{ 
  int npol = input->get_npol();
  if (npol==2)  return 4;
  else return 2;
}

/*! The quadrature components must be offset by one */
unsigned dsp::FadcTwoBitCorrection::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The in-phase and quadrature components must be interleaved */
unsigned dsp::FadcTwoBitCorrection::get_output_incr () const
{
  return 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::FadcTwoBitCorrection::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}


/*! MSB xQ xI yQ yI LSB

     idig   poln
      0   0re = xI
      1   0im = xQ
      2   1re = yI
      3   1im = yQ 
 */
unsigned
dsp::FadcTwoBitCorrection::get_shift (unsigned idig, unsigned isamp) const
{
  unsigned shift[4] = { 0, 2, 4, 6 }; // This should be right (I tested it)
  
  assert (isamp == 0 || isamp==1);
  assert (idig < 4);

  return shift[idig];
}

