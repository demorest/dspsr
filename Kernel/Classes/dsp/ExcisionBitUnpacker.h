//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Attic/ExcisionBitUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/09 05:51:16 $
   $Author: straten $ */

#ifndef __ExcisionBitUnpacker_h
#define __ExcisionBitUnpacker_h

#include "dsp/ExcisionUnpacker.h"
#include "dsp/BitUnpacker.h"

namespace dsp {

  //! Implements an ExcisionUnpacker using an existing BitUnpacker
  class ExcisionBitUnpacker: public ExcisionUnpacker
  {

  public:

    //! Null constructor
    ExcisionBitUnpacker (const char* name = "ExcisionBitUnpacker");

    //! Set the BitUnpacker to be used to unpack data
    void set_unpacker (BitUnpacker*);

  protected:

    //! Unpack a single digitized stream from raw into data
    virtual void dig_unpack (float* output_data,
			     const unsigned char* input_data, 
			     uint64 ndat,
			     unsigned digitizer,
			     unsigned* weights,
			     unsigned nweights);

    //! Unpacker implements unpack routine
    Reference::To<BitUnpacker> unpacker;

    virtual unsigned get_input_stride (unsigned nfloat_out) const;

    std::vector<char> number_of_low_states;

    unsigned minimum_ndat_per_weight;
  };

}

#endif
