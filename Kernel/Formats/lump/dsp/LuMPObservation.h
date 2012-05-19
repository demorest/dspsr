//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __LuMPObservation_h
#define __LuMPObservation_h

#include "dsp/ASCIIObservation.h"

namespace dsp {
 
  enum BinaryFormat
  {
    // Unknown format
    UnknownBinForm,
    // Integer
    IntegerBinForm,
    // IEEE floating point
    IEEE_FloatBinForm
  };

  enum DataEndianness
  {
    // Unknown endianness
    UnknownDataEndian,
    // Little endian
    DataLittleEndian,
    // Big endian
    DataBigEndian
  };

  enum PolOrdering
  {
    // When we don't know what is going on
    UnknownOrdering,
    // dspsr natural order for single dish
    // PP, QQ, Re[PQ], Im[PQ]
    DSPSROrdering,
    // Correlator Jones matrix natural ordering
    // PP PQ QP QQ
    JonesMatrixOrdering
  };

  //! Parses Observation attributes from a LuMP header
  class LuMPObservation : public ASCIIObservation {

  public:

    //! Construct from a LuMP file header
    LuMPObservation (const char* header);

    //! Set the binary format
    virtual void set_binary_format (BinaryFormat _binary_format)
    { binary_format = _binary_format; }
    //! Return the binary format
    BinaryFormat get_binary_format() const { return binary_format; }

    //! Set the data endianness
    virtual void set_data_endianness (DataEndianness _data_endianness)
    { data_endianness = _data_endianness; }
    //! Return the binary format
    DataEndianness get_data_endianness() const { return data_endianness; }

    //! Set the data endianness
    virtual void set_pol_ordering (PolOrdering _pol_ordering)
    { pol_ordering = _pol_ordering; }
    //! Return the binary format
    PolOrdering get_pol_ordering() const { return pol_ordering; }

  private:


    //! Format of binary data
    BinaryFormat binary_format;

    //! Endianness of the data
    DataEndianness data_endianness;

    //! Ordering of polarization arrays
    PolOrdering pol_ordering;

  };
  
}

#endif
