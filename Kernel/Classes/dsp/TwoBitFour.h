//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TwoBitFour.h

#ifndef __TwoBitFour_h
#define __TwoBitFour_h

#include "dsp/TwoBitLookup.h"

namespace dsp
{
  //! Unpack four 2-bit samples per byte from an array of bytes
  class TwoBitFour : public TwoBitLookup
  {

  public:

    static const unsigned samples_per_byte;
    static const unsigned lookup_block_size;

    //! Flag set when the data should be flagged as bad
    bool bad;

    //! Build the output value lookup table
    void lookup_build (TwoBitTable*, JenetAnderson98* = 0);

    //! Build counts of low voltage 2-bit states in each byte
    void nlow_build (TwoBitTable* table);

    //! Implement TwoBitLookup::get_lookup_block
    void get_lookup_block (float* lookup, TwoBitTable* table);

    //! Implement TwoBitLookup::get_lookup_block_size
    unsigned get_lookup_block_size ();

    template<class Iterator>
    inline void prepare (Iterator input, unsigned ndat)
    {
      const unsigned nbyte = ndat / samples_per_byte;
      unsigned total = 0;

      nlow = 0;

      for (unsigned bt=0; bt < nbyte; bt++)
      {
        nlow += nlow_lookup[ *input ];
	total += *input;
	++ input;
      }

      bad = (total == 0);
    }
    
    template<class Iterator>
    inline void unpack (Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& _nlow)
    {
      const unsigned nbyte = ndat / samples_per_byte;
      _nlow = nlow;

      // if data are complex, divide n_low by two
      nlow /= ndim;

      if (nlow < nlow_min)
	nlow = nlow_min;

      else if (nlow > nlow_max)
	nlow = nlow_max;
      
      float* lookup = lookup_base + (nlow-nlow_min) * lookup_block_size;

      for (unsigned bt=0; bt < nbyte; bt++)
      {
	float* fourval = lookup + *input * samples_per_byte;
	++ input;
	
	for (unsigned pt=0; pt < samples_per_byte; pt++)
        {
	  *output = fourval[pt];
	  output += output_incr;
	}
      }
    }
    
  protected:
    
    char nlow_lookup [256];
  };

}

#endif
