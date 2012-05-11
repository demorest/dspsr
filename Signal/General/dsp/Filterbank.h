//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Filterbank.h,v $
   $Revision: 1.26 $
   $Date: 2011/10/07 11:01:50 $
   $Author: straten $ */

#ifndef __Filterbank_h
#define __Filterbank_h

#include "dsp/Convolution.h"

namespace dsp {
  
  //! Breaks a single-band TimeSeries into multiple frequency channels
  /* This class implements the coherent filterbank technique described
     in Willem van Straten's thesis.  */

  class Filterbank: public Convolution {

  public:

    //! Configuration options
    class Config;

    //! Null constructor
    Filterbank (const char* name = "Filterbank", Behaviour type = outofplace);

    //! Prepare all relevant attributes
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64_t get_minimum_samples_lost () { return nsamp_overlap; }

    //! Set the number of channels into which the input will be divided
    void set_nchan (unsigned _nchan) { nchan = _nchan; }

    //! Get the number of channels into which the input will be divided
    unsigned get_nchan () const { return nchan; }

    unsigned get_nchan_subband () const {return nchan_subband; }

    //! Set the frequency resolution factor
    void set_freq_res (unsigned _freq_res) { freq_res = _freq_res; }
    void set_frequency_resolution (unsigned fres) { freq_res = fres; }

    //! Get the frequency resolution factor
    unsigned get_freq_res () const { return freq_res; } 
    unsigned get_frequency_resolution () const { return freq_res; }

    void set_frequency_overlap (unsigned over) { overlap_ratio = over; }
    unsigned get_frequency_overlap () const { return overlap_ratio; }

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Perform the filterbank step 
    virtual void filterbank ();
    virtual void custom_prepare () {}

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Frequency resolution factor
    unsigned freq_res;

    //! Frequency channel overlap ratio
    double overlap_ratio;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    void make_preparations ();
    void prepare_output (uint64_t ndat = 0, bool set_ndat = false);
    void resize_output (bool reserve_extra = false);

    unsigned nchan_subband;
  };
 
}

#endif

