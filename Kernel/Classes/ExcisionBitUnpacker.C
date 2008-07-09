/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionBitUnpacker.h"

    //! Null constructor
dsp::ExcisionBitUnpacker::ExcisionBitUnpacker (const char* name)
  : ExcisionUnpacker (name)
{
}

//! Set the BitUnpacker to be used to unpack data
void dsp::ExcisionBitUnpacker::set_unpacker (BitUnpacker* u)
{
  unpacker = u;
}

//! Get the offset (number of bytes) into input for the given digitizer
unsigned dsp::ExcisionBitUnpacker::get_input_offset (unsigned idig) const
{
}

//! Get the offset to the next byte containing the current digitizer data
unsigned dsp::ExcisionBitUnpacker::get_input_incr () const
{
}

//! Get the offset (number of floats) between consecutive digitizer samples
unsigned dsp::ExcisionBitUnpacker::get_output_incr () const
{
}

//! Unpack a single digitized stream from raw into data
void dsp::ExcisionBitUnpacker::dig_unpack (float* output_data,
					   const unsigned char* input_data, 
					   uint64 ndat,
					   unsigned digitizer,
					   unsigned* weights,
					   unsigned nweights)
{
}
