//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/SigProcDigitizer.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/01 10:18:44 $
   $Author: straten $ */

#ifndef __SigProcDigitizer_h
#define __SigProcDigitizer_h

#include "dsp/Digitizer.h"

namespace dsp
{  
  //! Converts floating point values to N-bit sigproc filterbank format
  class SigProcDigitizer: public Digitizer
  {
  public:

    //! Default constructor
    SigProcDigitizer ();

    //! Set the number of bits per sample
    void set_nbit (unsigned);

    //! Get the number of bits per sample
    unsigned get_nbit () const;

    //! Assumes all data are offset to zero mean and scaled to unit variance
    void pack ();

  protected:
    unsigned nbit;

  };
}

#endif
