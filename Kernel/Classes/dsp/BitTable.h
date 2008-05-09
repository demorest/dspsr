//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BitTable_h
#define __BitTable_h

#include "Reference.h"

namespace dsp {

  //! Look-up table for converting N-bit digitized to floating point numbers
  class BitTable : public Reference::Able
  {
  public:

    //! Number of bits per bytes
    static const unsigned bits_per_byte;

    //! Number of unique 8-bit combinations
    static const unsigned unique_bytes;

    enum Type
    {
      OffsetBinary,
      TwosComplement,
      SignMagnitude    // for 2-bit only
    };

    //! Constructor
    BitTable (unsigned nbit, Type type, bool reverse_bits = false);
    
    //! Destructor
    virtual ~BitTable ();

    //! Get the number of floating point values per byte, 8/N
    unsigned get_values_per_byte () const { return values_per_byte; }

    //! Get the number of unique values of an N-bit integer, 2^N
    unsigned get_unique_values () const { return unique_values; }

    //! Return the digitization convention
    Type get_type () const { return type; }

    //! Returns pointer to values_per_byte floats represented by byte
    const float* get_values (unsigned byte = 0);

    //! Generate a look-up table for conversion to floating point
    void generate (float* table) const;

    //! Generate a look-up table of unique_values floats
    virtual void generate_unique_values (float* values) const;

    //! Extract the ith sample from byte
    virtual unsigned extract (unsigned byte, unsigned i) const;

    //! Return the optimal variance of normally distributed samples
    virtual double get_optimal_variance ();

  protected:

    //! For each unsigned char, values of the two output voltage states
    float* table;

    //! Digitization convention
    Type type;

    //! Number of bits
    const unsigned nbit;

    //! Reverse the order of the bits
    const bool reverse_bits;

    //! Number of N-bit values per byte
    const unsigned values_per_byte;

    //! Number of unique N-bit values
    const unsigned unique_values;

    //! N-bit mask
    const unsigned nbit_mask;

    //! Build the lookup table
    void build ();

  };

}

#endif // !defined(__BitTable_h)
