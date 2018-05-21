/*

 */

#ifndef __dsp_UWBUnpacker_h
#define __dsp_UWBUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {
  
  class UWBUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    UWBUnpacker (const char* name = "UWBUnpacker");

    //! Destructor
    ~UWBUnpacker ();

    bool get_order_supported (TimeSeries::Order order) const;
    void set_output_order (TimeSeries::Order order);

    unsigned get_output_offset (unsigned idig) const;

    unsigned get_output_ipol (unsigned idig) const;

    unsigned get_output_ichan (unsigned idig) const;
    
    unsigned get_ndim_per_digitizer () const;

    //! Cloner (calls new)
    virtual UWBUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:
    
    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

    bool first_block;

  };

  class UWBUnpacker::Engine : public Reference::Able
  {
  public:

    virtual void unpack(const BitSeries * input, TimeSeries * output) = 0;

    virtual bool get_device_supported (Memory* memory) const = 0;

    virtual void set_device (Memory* memory) = 0;

    virtual void setup () = 0;

  };

}

#endif
