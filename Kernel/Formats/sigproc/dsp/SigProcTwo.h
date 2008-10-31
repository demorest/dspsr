//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/Attic/SigProcTwo.h,v $
   $Revision: 1.1 $
   $Date: 2008/10/31 06:33:36 $
   $Author: straten $ */

#ifndef __SigProcTwo_h
#define __SigProcTwo_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR files
  class SigProcTwo : public HistUnpacker 
  {

  public:
    
    //! Constructor
    SigProcTwo (const char* name = "SigProcTwo");

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

#endif // !defined(__SigProcTwo_h)
