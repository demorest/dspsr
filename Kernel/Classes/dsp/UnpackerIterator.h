//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/UnpackerIterator.h

#ifndef __UnpackerIterator_h
#define __UnpackerIterator_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Interface to Unpacker iterator implementations
  class Unpacker::Iterator
  {

  public:

    class Implementation;

    Iterator (Implementation* _impl);

    ~Iterator ();

    //! Dereferencing operator
    unsigned char operator * () const;
    
    //! Increment operator
    void operator ++ ();
    
    //! Comparison operator
    bool operator < (const unsigned char* ptr);

  protected:

    Implementation* impl;
  };

  //! Unpacker iterator implementation
  class Unpacker::Iterator::Implementation
  {

  public:

    virtual ~Implementation () {}

    //! Dereferencing operator
    virtual unsigned char get_value () const = 0;

    //! Increment operator
    virtual void increment () = 0;

    //! Comparison operator
    virtual bool less_than (const unsigned char*) = 0;

  };
}

#endif
