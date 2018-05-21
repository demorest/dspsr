/*

 */

#ifndef __dsp_MeerKATUnpacker_h
#define __dsp_MeerKATUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {
  
  class MeerKATUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    MeerKATUnpacker (const char* name = "MeerKATUnpacker");
    ~MeerKATUnpacker ();

    bool get_order_supported (TimeSeries::Order order) const;
    void set_output_order (TimeSeries::Order order);


    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;
    unsigned get_output_ichan (unsigned idig) const;

    //! Cloner (calls new)
    virtual MeerKATUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Engine used to unpack the data
    class Engine;

    void set_engine (Engine*);

  protected:

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

  private:

    bool device_prepared;

    int8_t * tfp_buffer;

    size_t tfp_buffer_size;

  };

  class MeerKATUnpacker::Engine : public Reference::Able
  {
  public:

    virtual void setup() = 0;

    virtual void unpack(float scale, const BitSeries * input, TimeSeries * output, unsigned sample_swap) = 0;

    virtual bool get_device_supported (Memory* memory) const = 0;

    virtual void set_device (Memory* memory) = 0;

  };

}

#endif
