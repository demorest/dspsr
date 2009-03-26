//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/Attic/SigProcOne.h,v $
   $Revision: 1.1 $
   $Date: 2009/03/26 03:00:15 $
   $Author: sixbynine $ */

#ifndef __SigProcOne_h
#define __SigProcOne_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR files
  class SigProcOne : public HistUnpacker 
  {

  public:
    
    //! Constructor
    SigProcOne (const char* name = "SigProcOne");

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

#endif // !defined(__SigProcOne_h)
