//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TimeSeries.h,v $
   $Revision: 1.20 $
   $Date: 2003/11/26 16:37:57 $
   $Author: wvanstra $ */

#ifndef __TimeSeries_h
#define __TimeSeries_h

#include <memory>

#include "Error.h"

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

    //! Copy constructor
    TimeSeries(const TimeSeries& ts);

    //! Called by constructor to initialise variables
    virtual void init();

    //! Cloner (calls new)
    virtual TimeSeries* clone();

    //! Swaps the two TimeSeries's.  Returns '*this'
    virtual TimeSeries& swap_data(TimeSeries& ts);

    //! Destructor
    virtual ~TimeSeries();

    //! Set this equal to copy
    virtual TimeSeries& operator = (const TimeSeries& copy);

    //! Add each value in data to this
    virtual TimeSeries& operator += (const TimeSeries& data);

    //! Multiple each value by this scalar
    virtual TimeSeries& operator *= (float mult);

    //! Copy the configuration of another TimeSeries instance (not the data)
    virtual void copy_configuration (const TimeSeries* copy);

    //! Hack together 2 different bands (not pretty)
    virtual void hack_together(vector<TimeSeries*> bands);

    //! Convenience interface to the above hack_together
    virtual void hack_together(TimeSeries* band1, TimeSeries* band2);

    //! Disable the set_nbit method of the Observation base class
    virtual void set_nbit (unsigned);

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (uint64 nsamples);

    //! Use the supplied array to store nsamples time samples.
    //! Always deletes existing data
    virtual void resize( uint64 nsamples, uint64 bytes_supplied, unsigned char* buffer);

    //! Equivalent to resize(0) but instead of deleting data, returns the pointer for reuse elsewhere
    virtual void zero_resize(unsigned char*& _buffer, uint64& nbytes);

    //! Offset the base pointer by offset time samples
    virtual void seek (int64 offset);

    //! Return pointer to the specified block of time samples
    virtual float* get_datptr (unsigned ichan=0,unsigned ipol=0);

    //! Return pointer to the specified block of time samples
    virtual const float* get_datptr (unsigned ichan=0,unsigned ipol=0) const;

    //! Return a pointer to the ichan'th frequency ordered channel and pol
    virtual float* get_ordered_datptr(unsigned ichan,unsigned ipol);

    //! Append little onto the end of 'this'
    //! If it is zero, then none were and we assume 'this' is full
    //! If it is nonzero, but not equal to little->get_ndat(), then 'this' is full too
    //! If it is equal to little->get_ndat(), it may/may not be full.
    virtual uint64 append (const TimeSeries* little);

    //! Only append the one chan/pol pair
    virtual uint64 append(const TimeSeries* little,unsigned ichan,unsigned ipol);

    //! Set all values to zero
    virtual void zero ();

    //! Return the mean of the data for the specified channel and poln
    double mean (unsigned ichan, unsigned ipol);

    //! Check that each floating point value is roughly as expected
    virtual void check (float min=-10.0, float max=10.0);

    //! Delete the current data buffer and attach to this one
    //! This is dangerous as it ASSUMES new data buffer has been pre-allocated and is big enough.  Beware of segmentation faults when using this routine.
    //! Also do not try to delete the old memory once you have called this- the TimeSeries::data member now owns it.
    virtual void attach(auto_ptr<float> _data);

    //! Call this when you do not want to transfer ownership of the array
    virtual void attach(float* _data);

    //! Copy from the back of 'tseries' into the front of 'this'
    //! This method is quite low-level- you have to fix up things like start_time, the observation data, ndat, and the subsize are all set correctly yourself
    //! Method basically just does lots of memcpys
    virtual void copy_from_back(TimeSeries* tseries, uint64 nsamples);

    //! Copy from the front of 'tseries' into the front of 'this'
    //! This method is quite low-level- you have to fix up things like start_time, the observation data, ndat, and the subsize are all set correctly yourself
    //! Method basically just does lots of memcpys
    virtual void copy_from_front(TimeSeries* tseries, uint64 nsamples);

    //! Check to see if subsize==ndat
    virtual bool full(){ return subsize==ndat; }

    //! Set ndat to subsize- the most it can be
    virtual void set_maximal_ndat(){ ndat = subsize; }

    //! Calculates the mean and the std dev of the timeseries, removes the mean, and scales to sigma
    virtual void normalise();

    //! Returns the maximum bin of channel 0, pol 0
    virtual unsigned get_maxbin();

    /*! This:
      (1) Puts the first 'this->ndat-ts->ndat' timesamples of 'this' into the last possible spots of 'this'
      (2) Copies 'ts' into the start of 'this'
      (3) So now the last few samples of 'this' are deleted and the timeseries starts earlier.
    */
    void rotate_backwards_onto(TimeSeries* ts);
    
    // Inquire the stride between floats representing the same timesample but in different chan/pol groups.  This is called by CoherentFBWriter 
    virtual uint64 get_subsize(){ return subsize; }

  protected:

    //! Called by append()
    void append_checks(uint64& ncontain,uint64& ncopy,
		       const TimeSeries* little);

    //! The data buffer
    float* buffer;

    //! The size of the data buffer 
    uint64 size;

    //! Pointer into buffer, offset to the first time sample requested by user
    float* data;

    //! The number of floats in a data sub-division
    uint64 subsize;
  };

  class TimeSeriesPtr{
    public:
    
    TimeSeries* ptr;
   
    TimeSeries& operator * () const
    { if(!ptr) throw Error(InvalidState,"dsp::TimeSeriesPtr::operator*()","You have called operator*() when ptr is NULL"); return *ptr; }
    TimeSeries* operator -> () const 
    { if(!ptr) throw Error(InvalidState,"dsp::TimeSeriesPtr::operator*()","You have called operator*() when ptr is NULL"); return ptr; }
        
    TimeSeriesPtr& operator=(const TimeSeriesPtr& tsp){ ptr = tsp.ptr; return *this; }

    TimeSeriesPtr(const TimeSeriesPtr& tsp){ operator=(tsp); }
    TimeSeriesPtr(TimeSeries* _ptr){ ptr = _ptr; }
    TimeSeriesPtr(){ ptr = 0; }

    bool operator < (const TimeSeriesPtr& tsp) const;

    ~TimeSeriesPtr(){ }
  };

  // This class just stores a pointer into a particular channel/polarisation pair's data
  class ChannelPtr{
  public :
    TimeSeries* ts;
    float* ptr;
    unsigned ichan;
    unsigned ipol;

    void init(TimeSeries* _ts,unsigned _ichan, unsigned _ipol)
    { ts=_ts; ichan=_ichan; ipol = _ipol; ptr=ts->get_datptr(ichan,ipol); }

    ChannelPtr& operator=(const ChannelPtr& c){ ts=c.ts; ptr=c.ptr; ichan=c.ichan; ipol=c.ipol; return *this; }

    ChannelPtr()
    { ts = 0; ptr = 0; ichan=ipol=0; }
    ChannelPtr(TimeSeries* _ts,unsigned _ichan, unsigned _ipol)
    { init(_ts,_ichan,_ipol); }
    ChannelPtr(TimeSeriesPtr _ts,unsigned _ichan, unsigned _ipol)
    { init(_ts.ptr,_ichan,_ipol); }
    ChannelPtr(const ChannelPtr& c){ operator=(c); }
    ~ChannelPtr(){ }
    
    bool operator < (const ChannelPtr& c) const;

    float& operator[](unsigned index){ return ptr[index]; }

    float get_centre_frequency(){ return ts->get_centre_frequency(ichan); }

  };
  
}
    
bool operator==(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2);
bool operator!=(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2);

#endif
