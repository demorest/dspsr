//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/gmrt/dsp/GMRTFilterbank16.h

#ifndef __GMRTFilterbank16_h
#define __GMRTFilterbank16_h

#include "dsp/Unpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR files
  class GMRTFilterbank16 : public Unpacker 
  {

  public:
    
    //! Constructor
    GMRTFilterbank16 (const char* name = "GMRTFilterbank16");

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

#endif // !defined(__GMRTFilterbank16_h)
