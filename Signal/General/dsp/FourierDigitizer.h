//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FourierDigitizer_h
#define __FourierDigitizer_h

#include "dsp/Digitizer.h"

class FourierDigitizer;

namespace dsp
{  
  //! Converts floating point values to N-bit Fourier filterbank format
  class FourierDigitizer: public Digitizer
  {
  public:

    //! Default constructor
    FourierDigitizer ();

    void set_nbit (int);

    //! Pack the data
    void pack ();

    //! Engine used to perform packing step
    class Engine;
    void set_engine (Engine*);

  private:
    
    int n_bit;

    Reference::To<Engine> engine;

  };

  class FourierDigitizer::Engine : public Reference::Able
  {
  public:
    virtual void pack (int nbit, const TimeSeries* in, BitSeries* out) = 0;

    virtual void finish () = 0;
  };

}

#endif
