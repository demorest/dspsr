//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitTable.h,v $
   $Revision: 1.4 $
   $Date: 2002/08/15 09:06:55 $
   $Author: wvanstra $ */


#ifndef __TwoBitTable_h
#define __TwoBitTable_h

namespace dsp {

  //! Look-up tables for conversion from two-bit to floating point numbers
  /*! 

  */
  class TwoBitTable {

  public:

    enum Type { Unset, OffsetBinary, SignMagnitude, TwosComplement };

    //! Constructor
    TwoBitTable (Type type);
    
    //! Destructor
    ~TwoBitTable ();

    //! Returns a pointer to the first of four values represented by one byte
    const float* get_four_vals (unsigned byte) const
    {
      return table + 4 * byte; 
    }

    //! Return the value of the low voltage state
    float get_lo_val () const { return lo_val; }

    //! Initialize a look-up table for byte to 4xfloating point conversion
    static void generate (float* table, Type type, float lo, float hi);

    //! Initialize a look-up table for 2-bit to floating point conversion
    static void four_vals (float* vals, Type type, float lo, float hi);

  protected:

    //! For each unsigned char, values of the four output voltage states
    float* table;

    //! Low voltage
    float lo_val;

  private:
    TwoBitTable () { /* allow no one to contruct without a type */ }

  };

}

#endif // !defined(__TwoBitTable_h)
