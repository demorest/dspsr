//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Filterbank.h,v $
   $Revision: 1.9 $
   $Date: 2008/04/08 04:32:44 $
   $Author: straten $ */

#ifndef __Filterbank_h
#define __Filterbank_h

#include "dsp/Convolution.h"
#include "FTransformAgent.h"

namespace dsp {
  
  //! Breaks a single-band TimeSeries into multiple frequency channels
  /* This class implements the coherent filterbank technique described
     in Willem van Straten's thesis.  */

  class Filterbank: public Convolution {

  public:

    //! Null constructor
    Filterbank ();

    //! Prepare all relevant attributes
    void prepare ();

    //! Get the minimum number of samples required for operation
    uint64 get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64 get_minimum_samples_lost () { return nsamp_overlap; }

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the number of channels into which the input will be divided
    void set_nchan (unsigned _nchan) { nchan = _nchan; }

    //! Get the number of channels into which the input will be divided
    unsigned get_nchan () const { return nchan; }

    //! Set the frequency resolution factor
    void set_freq_res (unsigned _freq_res) { freq_res = _freq_res; }
    void set_frequency_resolution (unsigned fres) { freq_res = fres; }

    //! Get the frequency resolution factor
    unsigned get_freq_res () const { return freq_res; } 

    //! Set the time resolution factor
    void set_time_res (unsigned _time_res) { time_res = _time_res; }

    //! Get the time resolution factor
    unsigned get_time_res () const { return time_res; } 

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Time resolution factor
    unsigned time_res;

    //! Frequency resolution factor
    unsigned freq_res;

  private:

    void make_preparations ();
    void prepare_output ();

    unsigned nchan_subband;
    unsigned nfilt_tot;
    unsigned nfilt_pos;
    unsigned nfilt_neg;

    unsigned nsamp_overlap;
    unsigned nsamp_fft;
    unsigned nsamp_step;
    unsigned nsamp_tres;

    double scalefac;

    bool matrix_convolution;

    FTransform::Plan* forward;
    FTransform::Plan* backward;

  };
  
}

#endif
