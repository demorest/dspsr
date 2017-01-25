//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/bpsr/dsp/BPSRUnpacker.h

#ifndef __BPSRUnpacker_h
#define __BPSRUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR files
  class BPSRUnpacker : public HistUnpacker 
  {

  public:
    
    //! Constructor
    BPSRUnpacker (const char* name = "BPSRUnpacker");

   protected:
    
    //! The unpacking routine
    virtual void unpack ();

    //! Return true; support any output order
    virtual bool get_order_supported (TimeSeries::Order) const;

    //! Set the order of the dimensions in the output TimeSeries
    virtual void set_output_order (TimeSeries::Order);

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    unsigned get_output_ichan (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;


  };

}

#endif // !defined(__BPSRUnpacker_h)
