//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitTable.h,v $
   $Revision: 1.6 $
   $Date: 2002/10/07 01:48:37 $
   $Author: wvanstra $ */


#ifndef __TwoBitTable_h
#define __TwoBitTable_h

#include "Reference.h"

namespace dsp {

  //! Look-up tables for conversion from two-bit to floating point numbers
  /*! 

  */
  class TwoBitTable : public Reference::Able {

  public:

    enum Type { Unset, OffsetBinary, SignMagnitude, TwosComplement };

    //! Constructor
    TwoBitTable (Type type);
    
    //! Destructor
    virtual ~TwoBitTable ();

    //! Build a two-bit table with the current attributes
    void build ();

    //! Returns a pointer to four floating-point values represented by one byte
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

  private:
    TwoBitTable () { /* allow no one to contruct without a type */ }

  };

}

#endif // !defined(__TwoBitTable_h)
