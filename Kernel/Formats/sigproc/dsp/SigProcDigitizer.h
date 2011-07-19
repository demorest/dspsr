//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/SigProcDigitizer.h,v $
   $Revision: 1.3 $
   $Date: 2011/07/19 14:59:41 $
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

    void set_nbit (int);

    //! Pack the data
    void pack ();

    //! Special case for floating point data
    void pack_float ();

  };
}

#endif
