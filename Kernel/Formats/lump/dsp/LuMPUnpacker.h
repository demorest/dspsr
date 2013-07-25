//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011, 2013 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __LuMPUnpacker_h
#define __LuMPUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp
{

  //! Unpack LUMP data (4, 8, 16, -16, -32, -64 bits per sample)
  // Use the basic Unpacker mother class, as
  // HistUnpacker has too many complications with conversion from float, etc.
  class LuMPUnpacker : public Unpacker 
  {

  public:
    
    //! Constructor
    LuMPUnpacker (const char* name = "LuMPUnpacker");

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

#endif // !defined(__LuMPUnpacker_h)
