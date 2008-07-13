//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitFour.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/13 00:38:54 $
   $Author: straten $ */

#ifndef __TwoBitFour_h
#define __TwoBitFour_h

#include "dsp/TwoBitTable.h"

class JenetAnderson98;

namespace dsp
{
  class TwoBitTable;

  //! Unpack four 2-bit samples per byte from an array of bytes
  class TwoBitFour
  {

  public:

    static const unsigned samples_per_byte;
    static const unsigned lookup_block_size;

    TwoBitFour ();
    ~TwoBitFour ();

    //! Set the minimum acceptable value of nlow
    void set_nlow_min (unsigned min);

    //! Set the maximum acceptable value of nlow
    void set_nlow_max (unsigned max);

    //! Build the nlow per byte lookup table
    void nlow_build (TwoBitTable* table);

    //! Build the output value lookup table
    void lookup_build (unsigned nsamp, unsigned ndim,
                       TwoBitTable*, JenetAnderson98* = 0);

    template<class Iterator>
    inline void prepare (Iterator input, unsigned ndat)
    {
      const unsigned nbyte = ndat / samples_per_byte;
      nlow = 0;
      for (unsigned bt=0; bt < nbyte; bt++)
      {
        nlow += nlow_lookup[ *input ];
	++ input;
      }
    }
    
    template<class Iterator>
    inline void unpack (Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& _nlow)
    {
      const unsigned nbyte = ndat / samples_per_byte;
      _nlow = nlow;

      // if data are complex, divide n_low by two
      nlow /= ndim_per_digitizer;

      if (nlow < nlow_min || nlow > nlow_max)
	return;
      
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

    unsigned nlow;
    unsigned nlow_min;
    unsigned nlow_max;

    unsigned ndim_per_digitizer;

    float* lookup_base;
 
    // delete lookup_base
    void destroy ();
   
  };

}

#endif
