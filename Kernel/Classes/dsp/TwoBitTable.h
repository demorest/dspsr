//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitTable.h,v $
   $Revision: 1.1 $
   $Date: 2002/08/06 23:31:19 $
   $Author: pulsar $ */


#ifndef __TwoBitTable_h
#define __TwoBitTable_h

namespace dsp {

  //! Look-up tables for conversion from two-bit to floating point numbers
  /*! 

  */
  class TwoBitTable {

  public:
    
    enum Type { OffsetBinary, SignMagnitude, TwosComplement };

    //! Constructor
    TwoBitTable (Type type);
    
    //! Destructor
    ~TwoBitTable ();

    //! Returns a pointer to the first of four values represented by one byte
    const float* get_4vals (unsigned char byte) const
    {
      return table + 4 * unsigned(byte); 
    }

  protected:

    //! For each unsigned char, values of the four output voltage states
    float* table;

  private:
    TwoBitTable () { /* allow no one to contruct without a type */ }

  };

}

#endif // !defined(__TwoBitTable_h)
