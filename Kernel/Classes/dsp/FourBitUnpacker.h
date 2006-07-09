//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Stephen Ord
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/FourBitUnpacker.h,v $
   $Revision: 1.3 $
   $Date: 2006/07/09 13:27:10 $
   $Author: wvanstra $ */

#ifndef __FourBitUnpacker_h
#define __FourBitUnpacker_h

#include <vector>

#include "dsp/Unpacker.h"

#include "environ.h"

namespace dsp {

  class FourBitTable;
  //! Converts BitSeries data from four-bit digitized to floating-point values
  /*! The conversion method is a simple shift and OR operation. It also assumes that there 
    is only one polarisation in the data  */

  class FourBitUnpacker: public Unpacker {

  public:

    //! Null constructor
    FourBitUnpacker (const char* name = "FourBitUnpacker");

    //! Virtual destructor
    virtual ~FourBitUnpacker ();

    //! Set the digitisation convention
    void set_table (FourBitTable* table);

    //! Get the digitisation convention
    const FourBitTable* get_table () const;

    //! Return a pointer to a new instance of the appropriate sub-class
    static FourBitUnpacker* create (const BitSeries& input);

    //! return the stats from nsample samples
    int64 stats (vector<double>& mean, vector<double>& var);

    //! Number of samples for stats
    unsigned nsamples;
			     
  protected:


    //! The four bit table generator  
    Reference::To<FourBitTable> table;
    
    //! Perform the bit conversion transformation on the input TimeSeries
    virtual void transformation ();

    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();

    //! Simple unpacker
    void simple_unpack(const unsigned char * in, uint64 ndat);

    //! What ipol to unpack into
    unsigned get_output_ipol ();
	
  };
}
#endif
