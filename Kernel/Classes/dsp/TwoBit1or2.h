//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TwoBit1or2.h

#ifndef __TwoBit1or2_h
#define __TwoBit1or2_h

// #define _DEBUG

#include "dsp/TwoBitLookup.h"
#include <vector>

namespace dsp
{
  class TwoBit1or2 : public TwoBitLookup
  {
  public:

    TwoBit1or2 ();
    ~TwoBit1or2 ();

    //! Build the output value lookup table
    void lookup_build (TwoBitTable*, JenetAnderson98* = 0);

    //! Build the nlow per byte lookup table
    void nlow_build (TwoBitTable* table);

    //! Unpack a block of unique samples, given the current table
    void get_lookup_block (float* lookup, TwoBitTable* table);

    //! Return the number of unique samples per block
    unsigned get_lookup_block_size ();

  protected:
    
    char nlow_lookup [4];

    std::vector<unsigned char> temp_values;
    void create ();
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
      const unsigned n2 = n >> 1;  // quick division by 2
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

    bool bad;

    template<class Iterator>
    inline void prepare (Iterator& input, unsigned ndat)
    {
      temp_values.resize (ndat);

#ifdef _DEBUG
      std::cerr << "TwoBit<" << N << ">::prepare ndat=" << ndat << std::endl;
#endif

      unpacker (input, mask, &(temp_values[0]), ndat);

      nlow = 0;

      unsigned total = 0;

      for (unsigned idat=0; idat < ndat; idat++)
      {
	nlow += nlow_lookup[ temp_values[idat] ];
	total += temp_values[idat];
      }

      bad = (total == 0);
    }
    
    template<class Iterator>
    inline void unpack (const Iterator& input, unsigned ndat, 
			float* output, unsigned output_incr, unsigned& _nlow)
    {
      _nlow = nlow;

      nlow /= ndim;

#ifdef _DEBUG
      std::cerr << "TwoBit<" << N << ">::unpack ndat=" << ndat 
                << " nlow=" << nlow << " nlow_min=" << nlow_min << std::endl;
#endif

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

