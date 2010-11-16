//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseLockedFilterbank.h,v $
   $Revision: 1.3 $
   $Date: 2010/11/16 13:57:56 $
   $Author: demorest $ */

#ifndef __baseband_dsp_PhaseLockedFilterbank_h
#define __baseband_dsp_PhaseLockedFilterbank_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/TimeDivide.h"

namespace Pulsar {
  class Predictor;
}

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

    //! Has a folding predictor been set?
    bool has_folding_predictor() const { return bin_divider.get_predictor(); }

    //! Get the predictor
    const Pulsar::Predictor* get_folding_predictor() const
      { return bin_divider.get_predictor(); }

    //! The phase divider
    TimeDivide bin_divider;

    //! Get pointer to the output
    PhaseSeries* get_result() const { return output; }

    //! Normalize the spectra by the hits array
    void normalize_output ();

    //! Reset the output
    void reset ();

    //! Finalize anything
    void finish ();

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

    //! Internal:  time to start processing data
    uint64_t idat_start;

    //! Internal:  number of samples to process
    uint64_t ndat_fold;

  };
  
}

#endif
