//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitTable.h,v $
   $Revision: 1.9 $
   $Date: 2004/06/07 00:31:05 $
   $Author: cwest $ */


#ifndef __TwoBitTable_h
#define __TwoBitTable_h

#include "Reference.h"

namespace dsp {

  //! Look-up table for conversion from two-bit to floating point numbers
  /*! 
    The TwoBitTable class can build a lookup up table of four floating point
    numbers per one-byte character.
   */
  class TwoBitTable : public Reference::Able {

  public:

    enum Type { Unset, 
		OffsetBinary, 
		SignMagnitude,
		TwosComplement 
    };

    //! Number of unique 8-bit combinations
    static const unsigned unique_bytes;

    //! Number of 2-bit values per byte
    static const unsigned vals_per_byte;

    //! Constructor
    TwoBitTable (Type type);
    
    //! Destructor
    virtual ~TwoBitTable ();

    //! Build a two-bit table with the current attributes
    void build ();

    //! Returns pointer to the four floating-point values represented by byte
    const float* get_four_vals (unsigned byte);

    //! Set the value of the low voltage state
    void set_lo_val (float lo_val);

    //! Return the value of the low voltage state
    float get_lo_val () const { return lo_val; }

    //! Set the value of the high voltage state
    void set_hi_val (float hi_val);

    //! Return the value of the high voltage state
    float get_hi_val () const { return hi_val; }

    //! Set the digitization convention
    void set_type (Type type);

    //! Return the digitization convention
    Type get_type () const { return type; }

    //! Set the flip value to be true or false
    void set_flip (bool flipped);

    //! Get the flip value
    bool get_flip () const { return flip; }

    //! Generate a look-up table for byte to 4xfloating point conversion
    void generate (float* table) const;

    //! Generate a look-up table for 2-bit to floating point conversion
    void four_vals (float* vals) const;

    //! Extract from byte the 2-bit number corresponding to sample
    virtual unsigned twobit (unsigned byte, unsigned sample) const;

  protected:

    //! For each unsigned char, values of the four output voltage states
    float* table;

    //! Flag that the lookup table has been built for specified attributes
    bool built;
    
    //! Value of voltage in low state
    float lo_val;

    //! Value of voltage in high state
    float hi_val;

    //! Digitization convention
    Type type;

    //! Flip the two data bits - that is SignMag becomes MagSign
    bool flip;

  private:

    //! Private default constructor ensures that type is specified
    TwoBitTable () { }

  };

}

#endif // !defined(__TwoBitTable_h)
