/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/SMROTwoBitCorrection.h"
#include "dsp/TwoBitTable.h"

bool dsp::SMROTwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "SMRO"
    && observation->get_nbit() == 2
    && observation->get_state() == Signal::Nyquist;
}

//! Null constructor
dsp::SMROTwoBitCorrection::SMROTwoBitCorrection ()
  : SubByteTwoBitCorrection ("SMROTwoBitCorrection")
{
  //threshold = 1.5;
#ifdef SIGN_MAG
  table = new TwoBitTable (TwoBitTable::SignMagnitude);
#endif
#ifdef OFFSET_BIN
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
#endif
  table->set_flip(true);
}

/*! SMRO has four digitizers: Potentially 4 channels */
unsigned dsp::SMROTwoBitCorrection::get_ndig () const
{
#ifdef CHAN8
  return 8;
#endif
#ifdef CHAN4
  return 4;
#endif
#ifdef CHAN2
  return 2;   // TWO CHANNELS
#endif
}

/*! Each 2-bit sample from each digitizer is packed into one byte */
unsigned dsp::SMROTwoBitCorrection::get_ndig_per_byte () const
{ 
#ifdef CHAN8
  return 4;
#endif
#ifdef CHAN4
  return 4;
#endif
#ifdef CHAN2
  return 2;   // TWO CHANNELS
#endif
}

