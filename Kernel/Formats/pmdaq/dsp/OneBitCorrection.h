//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __OneBitCorrection_h
#define __OneBitCorrection_h

class OneBitCorrection;

#include <vector>

#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "environ.h"

namespace dsp {

  //! Converts a TimeSeries from one-bit digitized to floating-point values
  //! Note that algorithm 1 loses ndat%MM samples (And algorithm 1 is only one coded/enabled as at 20 January 2004)
  class OneBitCorrection: public Unpacker {

  public:

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Null constructor
    OneBitCorrection (const char* name = "OneBitCorrection");

    //! Virtual destructor
    virtual ~OneBitCorrection ();

    //! Get the number of independent digitizers
    unsigned get_ndig () const;

    //! Inquire the first on-disk channel to load [0]
    unsigned get_first_chan(){ return first_chan; }
    //! Set the the first on-disk channel to load (0 or 256 at present for two filter observations) [0]
    //! 18 Sep 2005- Shouldn't need to call this- it should be auto-set now
    void set_first_chan(unsigned _first_chan){ first_chan = _first_chan; }

    //! Inquire the on-disk channel at which to stop loading [99999]
    unsigned get_end_chan(){ return end_chan; }
    //! Set the the on-disk channel at which to stop loading (e.g. 192 or 512) [99999]
    //! 18 Sep 2005- Shouldn't need to call this- it should be auto-set now
    void set_end_chan(unsigned _end_chan){ end_chan = _end_chan; }
    


  protected:
    
    //! First on-disk channel to load in [0]
    unsigned first_chan;
    //! Stop loading on-disk channels at this channel (i.e. load one before this channel but not this one) [99999]
    unsigned end_chan; 

    //! Perform the bit conversion transformation on the input TimeSeries
    virtual void transformation ();

    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();

    //! Return true if OneBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);



  private:

    //! Generate the lookup table
    void generate_lookup();

    //! Lookup table
    float lookup[256*8];

  };
  
}


#endif
