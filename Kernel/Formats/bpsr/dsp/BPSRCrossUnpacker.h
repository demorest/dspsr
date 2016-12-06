//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008-2014 by Willem van Straten & Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BPSRCrossUnpacker_h
#define __BPSRCrossUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp
{

  //! Simple 8-bit to float unpacker for the BPSR cross pol files
  class BPSRCrossUnpacker : public HistUnpacker 
  {

  public:
    
    //! Constructor
    BPSRCrossUnpacker (const char* name = "BPSRCrossUnpacker");

    //! only unpack PP & QQ inputs
    bool set_output_ppqq ();
  
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

    float gain_polx;
    float gain_pol1;
    float gain_pol2;
    unsigned ppqq_bw;

  private:

    float reference_gain;

    float ppqq_scale[2];

    float pq_scale;
    
  private:

    bool unpack_ppqq_only;

  };

}

#endif // !defined(__BPSRCrossUnpacker_h)
