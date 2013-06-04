/***************************************************************************
 *
 *   Copyright (C) 2013 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MOPSRUnpacker_h
#define __dsp_MOPSRUnpacker_h

#include "dsp/EightBitUnpacker.h"

namespace dsp {
  
  class MOPSRUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    MOPSRUnpacker (const char* name = "MOPSRUnpacker");
    ~MOPSRUnpacker ();

    //! Cloner (calls new)
    virtual MOPSRUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    //! Return true if support the output order
    bool get_order_supported (TimeSeries::Order order) const;

    //! Set the order of the dimensions in the output TimeSeries
    virtual void set_output_order (TimeSeries::Order);

    BitSeries staging;

    void * gpu_stream;

    void unpack_on_gpu ();

    unsigned get_resolution ()const ;

  private:

    bool device_prepared;

  };
}

#endif
