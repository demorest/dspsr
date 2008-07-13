//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBit1or2.h,v $
   $Revision: 1.2 $
   $Date: 2008/07/13 00:38:54 $
   $Author: straten $ */

#ifndef __TwoBit1or2_h
#define __TwoBit1or2_h

#include "dsp/TwoBitTable.h"

class JenetAnderson98;

namespace dsp
{
  class TwoBitTable;

  class TwoBit1or2
  {
  public:

    TwoBit1or2 ();
    ~TwoBit1or2 ();

    //! Set the minimum acceptable value of nlow
    void set_nlow_min (unsigned min);

    //! Set the maximum acceptable value of nlow
    void set_nlow_max (unsigned max);

    //! Build the nlow per byte lookup table
    void nlow_build (TwoBitTable* table);

    //! Build the output value lookup table
    void lookup_build (unsigned ndat, unsigned ndim,
                       TwoBitTable*, JenetAnderson98* = 0);

  protected:
    
    char nlow_lookup [4];

    unsigned nlow;
    unsigned nlow_min;
    unsigned nlow_max;

    unsigned ndim_per_digitizer;

    float* lookup_base;
     unsigned char* temp_values;

    // delete temp_values and lookup_base
    void destroy ();

  };


  //! Defines interface to 2-bit extractors
  template<unsigned N>
  class TwoBitToChar
  {
  };

  //! Extracts one 2-bit number per byte
  template<>
  class TwoBitToChar<1>
  {
  public:
    template<class Iterator, class Mask>
    inline void operator() (Iterator& from, Mask& mask,
			    unsigned char* to, unsigned n)
    {
      for (unsigned i=0; i<n; i++)
      {
	to[i] = mask (*from);
	++ from;
      }
    }
  };

  //! Extracts two 2-bit numbers per byte
  template<>
  class TwoBitToChar<2>
  {
  public:
    template<class Iterator, class Mask>
    inline void operator() (Iterator& from, Mask& mask,
			    unsigned char* to, unsigned n)
    {
      const unsigned n2 = n/2;
      for (unsigned i=0; i<n2; i++)
      {
	to[i*2] = mask (*from, 0);
	to[i*2+1] = mask (*from, 1);
	++ from;
      }
    }
  };

  //! Unpack one or two 2-bit samples per byte from an array of bytes
  /*!
    \param N number of samples per byte
    \param Mask delivers the samples
  */
  template<unsigned N, class Mask>
  class TwoBit : public TwoBit1or2
  {

    TwoBitToChar<N> unpacker;

  public:

    Mask mask;

    static const unsigned samples_per_byte;
    static const unsigned lookup_block_size;

    template<class Iterator>
    inline void prepare (Iterator& input, unsigned ndat)
    {
#ifdef _DEBUG
      std::cerr << "TwoBit<" << N << ">::prepare ndat=" << ndat 
                << " temp=" << (void*) temp_values << std::endl;
#endif

      unpacker (input, mask, temp_values, ndat);

      nlow = 0;
      for (unsigned idat=0; idat < ndat; idat++)
	nlow += nlow_lookup[ temp_values[idat] ];
    }
    
    template<class Iterator>
    inline void unpack (const Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& _nlow)
    {
      _nlow = nlow;

      nlow /= ndim_per_digitizer;
      
      if (nlow < nlow_min || nlow > nlow_max)
	return;
      
      float* fourval = lookup_base + (nlow-nlow_min) * lookup_block_size;

      for (unsigned idat=0; idat < ndat; idat++)
	output[idat*output_incr] = fourval[ temp_values[idat] ];
    }
   
  };

}

template<unsigned N, class Mask>
const unsigned dsp::TwoBit<N,Mask>::samples_per_byte = N;

template<unsigned N, class Mask>
const unsigned dsp::TwoBit<N,Mask>::lookup_block_size = 4;

#endif
