/*

 */

#ifndef __dsp_SKA1Unpacker_h
#define __dsp_SKA1Unpacker_h

//#define SKA1_ENGINE_IMPLEMENTATION

#include "dsp/EightBitUnpacker.h"
#include "ThreadContext.h"

namespace dsp {
  
  class SKA1Unpacker : public HistUnpacker
  {
  public:

    //! Constructor
    SKA1Unpacker (const char* name = "SKA1Unpacker");

    //! Destructor
    ~SKA1Unpacker ();

    //! Cloner (calls new)
    virtual SKA1Unpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

#ifdef SKA1_ENGINE_IMPLEMENTATION
    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);
#else
    void unpack_on_gpu ();
#endif

  protected:
    
#ifdef SKA1_ENGINE_IMPLEMENTATION
    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;
#else
    void * gpu_stream;
#endif

    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    //BitSeries staging;
    //unsigned get_resolution () const ;

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

  };

#ifdef SKA1_ENGINE_IMPLEMENTATION

  class SKA1Unpacker::Engine : public Reference::Able
  {
  public:
    virtual void unpack(float scale, const BitSeries * input, TimeSeries * output) = 0;

    virtual bool get_device_supported (Memory* memory) const = 0;

    virtual void set_device (Memory* memory) = 0;

  };

#endif

}

#endif
