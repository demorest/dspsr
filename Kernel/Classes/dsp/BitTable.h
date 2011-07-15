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

    //! Interpretation of the bits in each value
    enum Type
    {
      OffsetBinary,
      TwosComplement,
      SignMagnitude    // for 2-bit only
    };

    //! The order of values in each byte (bit significance)
    enum Order
    {
      MostToLeast,
      LeastToMost
    };

    //! Constructor
    BitTable (unsigned nbit, Type type, bool reverse_bits = false);
    
    //! Destructor
    virtual ~BitTable ();

    //! Set the effective number of bits
    /*! For example, a digitizer may set 8-bit thresholds to effect a
      6-bit digitizer, leaving head room for RFI */
    void set_effective_nbit (unsigned bits);
    unsigned get_effective_nbit () const { return effective_nbit; }

    //! Set the order of the samples in each byte
    void set_order (Order);
    Order get_order () const { return order; }

    //! Get the number of floating point values per byte, 8/N
    unsigned get_values_per_byte () const { return values_per_byte; }

    //! Get the number of unique values of an N-bit integer, 2^N
    unsigned get_unique_values () const { return unique_values; }

    //! Get the scale factor used to normalize the variance to unity
    double get_scale () const;

    //! Return the digitization convention
    Type get_type () const { return type; }

    //! Returns pointer to values_per_byte floats represented by byte
    const float* get_values (unsigned byte = 0) const;

    //! Generate a look-up table for conversion to floating point
    void generate (float* table) const;

    //! Generate a look-up table of unique_values floats
    virtual void generate_unique_values (float* values) const;

    //! Extract the ith sample from byte
    virtual unsigned extract (unsigned byte, unsigned i) const;

    //! Return the optimal variance of normally distributed samples
    virtual double get_optimal_variance () const;

    //! Return the optimal threshold closest to and less than unity
    virtual double get_nlow_threshold () const;

    //! Return the number of low voltage states in each of 256 bytes
    virtual void get_nlow_lookup (char* nlow_lookup) const;

  protected:

    //! For each unsigned char, values of the two output voltage states
    float* table;

    //! Digitization convention
    Type type;

    //! Number of bits
    const unsigned nbit;

    //! The effective number of bits
    unsigned effective_nbit;

    //! Reverse the order of the bits
    const bool reverse_bits;

    //! The order of the samples in each byte
    Order order;

    //! Number of N-bit values per byte
    const unsigned values_per_byte;

    //! Number of unique N-bit values
    const unsigned unique_values;

    //! N-bit mask
    const unsigned nbit_mask;

    //! The scale factor used to normalize the variance to unity
    mutable double scale;

    //! Build the lookup table
    void build ();

    //! Destroy the lookup table
    void destroy ();
  };

}

#endif // !defined(__BitTable_h)
