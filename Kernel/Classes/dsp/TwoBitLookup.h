//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TwoBitLookup.h

#ifndef __TwoBitLookup_h
#define __TwoBitLookup_h

#include "dsp/TwoBitTable.h"

class JenetAnderson98;

namespace dsp
{
  class TwoBitTable;

  //! Creates lookup table for 2-bit dynamic output level setting
  class TwoBitLookup
  {

  public:

    TwoBitLookup ();
    virtual ~TwoBitLookup ();

    //! Set the number of time samples in each population counted
    void set_ndat (unsigned ndat);

    //! Set the dimension of the time samples (1=real, 2=complex)
    void set_ndim (unsigned ndim);

    //! Set the minimum acceptable number of low voltage states
    void set_nlow_min (unsigned min);

    //! Set the maximum acceptable number of low voltage states
    void set_nlow_max (unsigned max);

    //! Build the output value lookup table
    virtual void lookup_build (TwoBitTable*, JenetAnderson98* = 0);

    //! Unpack a block of unique samples, given the current table
    virtual void get_lookup_block (float* lookup, TwoBitTable* table) = 0;

    //! Return the number of unique samples per block
    virtual unsigned get_lookup_block_size () = 0;

  protected:
    
    unsigned nlow;
    unsigned nlow_min;
    unsigned nlow_max;

    unsigned ndat;
    unsigned ndim;

    float* lookup_base;
 
    // create lookup_base
    virtual void create ();

  private:

    // delete lookup_base
    void destroy ();
  };

}

#endif
