//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Filterbank.h,v $
   $Revision: 1.4 $
   $Date: 2002/11/03 21:51:49 $
   $Author: wvanstra $ */

#ifndef __Filterbank_h
#define __Filterbank_h

#include "dsp/Convolution.h"

namespace dsp {
  
  //! Breaks a single-band Timeseries into multiple frequency channels
  /* This class implements the coherent filterbank technique described
     in Willem van Straten's thesis.  */

  class Filterbank: public Convolution {

  public:

    //! Null constructor
    Filterbank ();

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

    //! Get the frequency resolution factor
    unsigned get_freq_res () const { return freq_res; } 

    //! Set the time resolution factor
    void set_time_res (unsigned _time_res) { time_res = _time_res; }

    //! Get the time resolution factor
    unsigned get_time_res () const { return time_res; } 

  protected:

    //! Perform the convolution operation on the input Timeseries
    virtual void operation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Time resolution factor
    unsigned time_res;

    //! Frequency resolution factor
    unsigned freq_res;

  };
  
}

#endif
