/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mark5TwoBitCorrection.h"
#include "dsp/Mark5File.h"

#include "dsp/StepIterator.h"
#include "dsp/excision_unpack.h"

#include "vlba_stream.h"

bool dsp::Mark5TwoBitCorrection::matches (const Observation* observation)
{
  return can_do( observation );
}

bool dsp::Mark5TwoBitCorrection::can_do (const Observation* observation)
{
  return observation->get_machine() == "Mark5" 
    && observation->get_nbit() == 2
    && observation->get_npol() == 2
    && observation->get_nchan() == 4;
}

//! Null constructor
dsp::Mark5TwoBitCorrection::Mark5TwoBitCorrection ()
  : SubByteTwoBitCorrection ("Mark5TwoBitCorrection")
{
  bool reverse_bits = true;
  table = new TwoBitTable (TwoBitTable::OffsetBinary, reverse_bits);
  set_ndig (8);
  file = 0;
}

/*! Two 2-bit samples from two digitizers are packed into one byte */
unsigned dsp::Mark5TwoBitCorrection::get_ndig_per_byte () const
{ 
  return 2;
}

/*! 
  CHAN a = 000 = ipol 0
  CHAN b = 001 = ipol 0
  CHAN c = 010 = ipol 1
  CHAN d = 011 = ipol 1
  CHAN e = 100 = ipol 0
  CHAN f = 101 = ipol 0
  CHAN g = 110 = ipol 1
  CHAN h = 111 = ipol 1
*/
unsigned dsp::Mark5TwoBitCorrection::get_output_ipol (unsigned idig) const
{
  return (idig >> 1) & 1;
}

/*! 
  CHAN a = 000 = ichan 0
  CHAN b = 001 = ichan 2
  CHAN c = 010 = ichan 0
  CHAN d = 011 = ichan 2
  CHAN e = 100 = ichan 1
  CHAN f = 101 = ichan 3
  CHAN g = 110 = ichan 1
  CHAN h = 111 = ichan 3
*/
unsigned dsp::Mark5TwoBitCorrection::get_output_ichan (unsigned idig) const
{
  return ((idig << 1) & 2) | ((idig >> 2) & 1);
}

/*! Each input word contains two digitizers */
unsigned dsp::Mark5TwoBitCorrection::get_input_offset (unsigned idig) const 
{
  return idig / 2;
}

/*! Each input word is four bytes long */
unsigned dsp::Mark5TwoBitCorrection::get_input_incr () const 
{
  return 4;
}

/*!
  See Table 12 of 
  "Mark IIIA/IV/VLBA Tape Formats, Recording Modes and Compatibility"
  Revision 1.21, Alan R. Whitney, MIT Haystack Observatory, 10 June 2005

  Note that the bit numbers in this table run from 2 to 33.
  Subtract 2 from each bit to match the convention used in this method.

  Note: in 2bit32 mode, with fanout=2,
  Walter's code and the Mark5Unpacker produced the following:

  f=fanout
  c=channel
  s=LSB
  m=MSB             Table 12    Mark5Unpacker

  f=0 c=0 s=0  m=4  (CHAN a) - ichan 0 - ipol 0
  f=0 c=1 s=8  m=12 (CHAN c) - ichan 0 - ipol 1
  f=0 c=2 s=16 m=20 (CHAN e) - ichan 1 - ipol 0
  f=0 c=3 s=24 m=28 (CHAN g) - ichan 1 - ipol 1
  f=0 c=4 s=1  m=5  (CHAN b) - ichan 2 - ipol 0
  f=0 c=5 s=9  m=13 (CHAN d) - ichan 2 - ipol 1
  f=0 c=6 s=17 m=21 (CHAN f) - ichan 3 - ipol 0
  f=0 c=7 s=25 m=29 (CHAN h) - ichan 3 - ipol 1
  f=1 c=0 s=2  m=6
  f=1 c=1 s=10 m=14
  f=1 c=2 s=18 m=22
  f=1 c=3 s=26 m=30
  f=1 c=4 s=3  m=7
  f=1 c=5 s=11 m=15
  f=1 c=6 s=19 m=23
  f=1 c=7 s=27 m=31

 */

void dsp::Mark5TwoBitCorrection::dig_unpack (const unsigned char* input_data,
					       float* output_data,
					       uint64_t nfloat,
					       unsigned long* hist,
					       unsigned* weights,
					       unsigned nweights)
{
  if (!file)
  {
    file = get_Input<Mark5File>();
    if (!file)
      throw Error (InvalidState, "dsp::Mark5Unpacker::unpack",
		   "Input is not a Mark5File");
  }

  StepIterator<const unsigned char> iterator (input_data);
  iterator.set_increment ( get_input_incr() );

  struct VLBA_stream* vlba_stream = (struct VLBA_stream*) file->stream;

  // the byte pattern repeats every two digitizers
  unsigned channel = current_digitizer % 2;

  // CHAN b in Walter's code == channel 4
  gather.mask.shift0[0] = vlba_stream->basebits[channel * 4];

  // m in Walter's code
  // NOTE: -1 so that GatherMask::bitshift does not have to << before |
  gather.mask.shift1[0] = gather.mask.shift0[0] + 2*vlba_stream->fanout -1;

  // +2*f in Walter's code
  gather.mask.shift0[1] = gather.mask.shift0[0] + 2;  // s
  gather.mask.shift1[1] = gather.mask.shift1[0] + 2;  // m

  ExcisionUnpacker::excision_unpack (gather, iterator,
				     output_data, nfloat,
                                     hist, weights, nweights);
}
