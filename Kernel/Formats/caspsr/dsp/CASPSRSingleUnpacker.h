/*

 */

#ifndef __dsp_CASPSRSingleUnpacker_h
#define __dsp_CASPSRSingleUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {
  
  class CASPSRSingleUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    CASPSRSingleUnpacker (const char* name = "CASPSRSingleUnpacker");
    ~CASPSRSingleUnpacker ();

    //! Cloner (calls new)
    virtual CASPSRSingleUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    void unpack_default ();

    void unpack (uint64_t ndat, const unsigned char* from, 
		             float* into, const unsigned fskip,
		             unsigned long* hist);

    void * gpu_stream;

    void unpack_on_gpu ();

    unsigned get_resolution ()const ;

  private:

    bool device_prepared;

    //! maximum number of GPU threads per block
    int threadsPerBlock;

  };
}

#endif
