//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseLockedFilterbank.h,v $
   $Revision: 1.2 $
   $Date: 2006/07/09 13:27:13 $
   $Author: wvanstra $ */

#ifndef __baseband_dsp_PhaseLockedFilterbank_h
#define __baseband_dsp_PhaseLockedFilterbank_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/TimeDivide.h"

namespace dsp {
  
  //! Performs FFT at specific pulse phase windows
  /*! This class is particularly useful when maximum frequency resolution
    is required in dynamic spectra. */
  class PhaseLockedFilterbank 
    : public Transformation <TimeSeries, PhaseSeries> {

  public:

    //! Default constructor
    PhaseLockedFilterbank ();

    //! Set the number of channels into which the input will be divided
    void set_nchan (unsigned nchan);

    //! Get the number of channels into which the input will be divided
    unsigned get_nchan () const { return nchan; }

    //! Set the number of pulse phase windows in which to compute spectra
    void set_nbin (unsigned nbin);

    //! Get the number of pulse phase windows in which to compute spectra
    unsigned get_nbin () const { return nbin; }

    //! The phase divider
    TimeDivide divider;

    //! Normalize the spectra by the hits array
    void normalize_output ();

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Number of pulse phase windows in which to compute spectra
    unsigned nbin;

    //! Flag set when built
    bool built;

    //! Prepare internal variables
    void prepare ();

  };
  
}

#endif
