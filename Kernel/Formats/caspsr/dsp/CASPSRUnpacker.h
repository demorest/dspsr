/*

 */

#ifndef __dsp_CASPSRUnpacker_h
#define __dsp_CASPSRUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {
  
  class CASPSRUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    CASPSRUnpacker (const char* name = "CASPSRUpacker");

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    void unpack (uint64_t ndat,
		 const unsigned char* from, const unsigned nskip,
		 float* into, const unsigned fskip,
		 unsigned long* hist);

    BitSeries staging;
    void* gpu_stream;
    void unpack_on_gpu ();

    unsigned get_resolution ()const ;
  };
}

#endif
