//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TestInput.h,v $
   $Revision: 1.1 $
   $Date: 2003/08/20 09:17:41 $
   $Author: wvanstra $ */

#ifndef __TestInput_h
#define __TestInput_h

#include "environ.h"

namespace dsp {

  class BitSeries;
  class Input;

  //! A class used for testing in both test_Input.C and test_MPIRoot.C
  /*! This class serves no other purpose than to test the resolution
    attribute of the Input class in a variety of circumstances. */

  class TestInput {

  public:
    
    static bool verbose;

    //! Constructor
    TestInput ();

    //! Destructor
    ~TestInput ();
    
    //! Run test using two Input instances that refer to the same data
    void runtest (Input* input_a, Input* input_b);

    //! Get the number of errors encountered during runtest
    unsigned get_errors () const { return errors; }

    //! Return the number of time samples to load on each load_block
    uint64 get_block_size () const { return block_size; }
    //! Set the number of time samples to load on each load_block
    void set_block_size (uint64 _size) { block_size = _size; }
    
  protected:

    //! The number of time samples to load on each load_block
    uint64 block_size;

    //! The number of errors in runtest
    unsigned errors;

  };

}

#endif // !defined(__TestInput_h)
