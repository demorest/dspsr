//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Response.h,v $
   $Revision: 1.1 $
   $Date: 2002/08/20 04:58:16 $
   $Author: wvanstra $ */

#ifndef __Response_h
#define __Response_h

#include <vector>
#include <complex>

#include "Shape.h"
#include "Jones.h"

namespace dsp {

  class Timeseries;

  class Response : public Shape {

  public:

    //! null constructor
    Response();

    //! construct frequency response from complex phasors
    void set (const vector<complex<float> >& phasors);

    //! construct frequency response from jones matrices
    void set (const vector<Jones<float> >& jones);

    //! Multiply spectrum by complex frequency response
    void operate (float* spectrum, unsigned poln=0, unsigned chan=0);

    //! Multiply spectrum vector by complex matrix frequency response
    void operate (float* spectrum1, float* spectrum2, unsigned chan=0);

    //! Integrate the power of spectrum into self
    void integrate (float* spectrum, unsigned poln=0, unsigned chan=0);

    //! Integrate coherency matrix of vector spectrum into self
    void integrate (float* spectrum1, float* spectrum2, unsigned chan=0);

    //! Match the frequency response to the input Timeseries
    virtual void match (const Timeseries* input, unsigned nchan=0);

    //! Re-organized frequency bins to reflect natural ordering (DC->Nyq)
    void naturalize ();

  protected:

    //! Swap halves of bandpass(es)
    void swap (bool each_chan = true);

    //! Toggled every time Response::swap(false) is called (default: false)
    bool whole_swapped;

    //! Toggled every time Response::swap(true) is called (default: false)
    bool chan_swapped;

    //! Toggled when Response::rotate
    bool chan_shifted;

  };

}

#endif
