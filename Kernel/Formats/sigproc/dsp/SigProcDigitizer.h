//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/sigproc/dsp/SigProcDigitizer.h

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

    //! Set the manual scale factor
    void set_scale(float _scale) { scale_fac = _scale; }

    //! Set whether or not to apply the nbit-depedent scalings
    void use_digi_scales (bool _rescale) { rescale = _rescale; }

  protected:

    //! Additional scale factor to apply to data
    float scale_fac;

    //! Should the data be rescaled using the nbit-dependent values?
    bool rescale;

  };
}

#endif
