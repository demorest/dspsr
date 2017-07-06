//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/GenericEightBitUnpacker.h

#ifndef __GenericEightBitUnpacker_h
#define __GenericEightBitUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {

  //! Simple 8-bit to float unpacker for generic 8-bit files
  class GenericEightBitUnpacker : public EightBitUnpacker
  {

  public:
    
    //! Constructor
    GenericEightBitUnpacker ();

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

   protected:

    //! Return true if this unpacker can convert the Observation
    virtual bool matches (const Observation* observation);

    //! Override BitUnpacker::unpack
    virtual void unpack ();

    BitSeries staging;
    void* gpu_stream;
    void unpack_on_gpu ();
  };

}

#endif // !defined(__GenericEightBitUnpacker_h)

