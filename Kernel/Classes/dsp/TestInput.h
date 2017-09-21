//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TestInput.h

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
    uint64_t get_block_size () const { return block_size; }
    //! Set the number of time samples to load on each load_block
    void set_block_size (uint64_t _size) { block_size = _size; }
    
  protected:

    //! The number of time samples to load on each load_block
    uint64_t block_size;

    //! The number of errors in runtest
    unsigned errors;

  };

}

#endif // !defined(__TestInput_h)
