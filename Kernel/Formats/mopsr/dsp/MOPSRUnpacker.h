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

    unsigned get_output_offset (unsigned idig) const;

    unsigned get_output_ipol (unsigned idig) const;

    unsigned get_output_ichan (unsigned idig) const;

    unsigned get_ndim_per_digitizer () const;

    //! Cloner (calls new)
    virtual MOPSRUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! synch with the Input resolution
     void match_resolution (const Input*);

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

    unsigned get_resolution ()const;

    void unpack_on_gpu ();

  private:

    void validate_transformation();

    enum DataOrder {
      //! unknown input order
      NONE,
      //! PFB single antenna input
      TF,
      //! PFB multi antenna input
      FT,
      //! Beam Formed single antenna input
      T
    };

    DataOrder input_order;

    bool device_prepared;

    unsigned input_resolution;

    int debugd;

  };
}

#endif
