//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/sigproc/dsp/SigProcUnpacker.h

#ifndef __SigProcUnpacker_h
#define __SigProcUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Unpack SIGPROC data (1, 2, 4, or 8 bits per sample)
  class SigProcUnpacker : public HistUnpacker 
  {

  public:
    
    //! Constructor
    SigProcUnpacker (const char* name = "SigProcUnpacker");

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

#endif // !defined(__SigProcUnpacker_h)
