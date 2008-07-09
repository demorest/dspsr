//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/ExcisionUnpacker.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/09 02:59:11 $
   $Author: straten $ */

#ifndef __ExcisionUnpacker_h
#define __ExcisionUnpacker_h

#include "dsp/BitUnpacker.h"

namespace dsp {

  //! Excises digitized data with statistics outside of acceptable limits
  class ExcisionUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    ExcisionUnpacker (const char* name = "ExcisionUnpacker");

    //! Get the offset (number of bytes) into input for the given digitizer
    virtual unsigned get_input_offset (unsigned idig) const;

    //! Get the offset to the next byte containing the current digitizer data
    virtual unsigned get_input_incr () const;

    //! Get the offset (number of floats) between consecutive digitizer samples
    virtual unsigned get_output_incr () const;

    //! Set the number of time samples used to estimate undigitized power
    void set_ndat_per_weight (unsigned ndat_per_weight);

    //! Set the number of states in the histogram
    void set_nstate (unsigned nstate) { set_ndat_per_weight (nstate); }

    //! Set the cut off power for impulsive interference excision
    void set_cutoff_sigma (float cutoff_sigma);
    
    //! Get the cut off power for impulsive interference excision
    float get_cutoff_sigma() const { return cutoff_sigma; }

  protected:

    //! Unpack using dig_unpack then perform excision
    void unpack ();

    //! Unpack a single polarization from raw into data
    virtual void dig_unpack (float* output_data,
			     const unsigned char* input_data, 
			     uint64 ndat,
			     unsigned digitizer,
			     unsigned* weights = 0,
			     unsigned nweights = 0);

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of low states in ndat_per_weight points
    unsigned nlow_min;

    //! Maximum number of low states in ndat_per_weight points
    unsigned nlow_max;

    //! Lookup table and histogram dimensions reflect the attributes
    bool built;

  };
}
#endif
