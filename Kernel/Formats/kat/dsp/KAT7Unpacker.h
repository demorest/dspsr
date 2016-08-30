/*

 */

#ifndef __dsp_KAT7Unpacker_h
#define __dsp_KAT7Unpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {
  
  class KAT7Unpacker : public HistUnpacker
  {
  public:

    //! Constructor
    KAT7Unpacker (const char* name = "KAT7Unpacker");
    ~KAT7Unpacker ();

    //! Cloner (calls new)
    virtual KAT7Unpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    void unpack (uint64_t ndat, const unsigned char* from, 
		             float* into, const unsigned fskip,
		             unsigned long* hist);

    BitSeries staging;
    void * gpu_stream;
    void unpack_on_gpu ();

    unsigned get_resolution ()const ;

  private:

    bool device_prepared;

  };
}

#endif
