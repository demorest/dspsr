//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TimeSeries.h,v $
   $Revision: 1.3 $
   $Date: 2002/11/10 17:54:35 $
   $Author: wvanstra $ */

#ifndef __TimeSeries_h
#define __TimeSeries_h

#include "dsp/Observation.h"

namespace dsp {
  
  //! Container of dimension/time-major order floating point data.
  /* The TimeSeries class contains floating point data arranged as a
     function of frequency, polarization, time, and dimension, ie.

     f0p0t0d0...f0p0t0dD,f0p0t1d0...f0p0t1dD...f0p0tTd0...f0p0tTdD, 
     f0p1t0d0...f0p1t0dD,f0p1t1d0...f0p1t1dD...f0p1tTd0...f0p1tTdD, 
     ...
     f0pPt0d0...f0pPt0dD,f0pPt1d0...f0pPt1dD...f0pPtTd0...f0pPtTdD, 
     f1p0t0d0...f1p0t0dD,f1p0t1d0...f1p0t1dD...f1p0tTd0...f1p0tTdD, 
     ...
     fFpPt0d0...fFpPt0dD,fFpPt1d0...fFpPt1dD...fFpPtTd0...fFpPtTdD
  */
  class TimeSeries : public Observation {

  public:
    //! Null constructor
    TimeSeries ();

    //! Set this equal to copy
    virtual TimeSeries& operator = (const TimeSeries& copy);

    //! Add each value in data to this
    virtual TimeSeries& operator += (const TimeSeries& data);

    //! Disable set_nbit
    void set_nbit (unsigned);

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (uint64 nsamples);
    
    //! Return pointer to the specified block of time samples
    virtual float* get_datptr (unsigned ichan=0,unsigned ipol=0);

    //! Return pointer to the specified block of time samples
    virtual const float* get_datptr (unsigned ichan=0,unsigned ipol=0) const;

    //! Append little onto the end of this
    virtual void append (const TimeSeries* little);

    //! Set all values to zero
    virtual void zero ();

    //! Check that each floating point value is roughly as expected
    virtual void check (float min=-10.0, float max=10.0);

  protected:
    //! The data buffer
    float* data;

    //! The number of floats of the data buffer 
    uint64 size;

    //! The number of floats in a data sub-division
    uint64 subsize;

  };
  
}

#endif
