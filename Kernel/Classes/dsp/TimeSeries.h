//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TimeSeries.h,v $
   $Revision: 1.41 $
   $Date: 2007/11/15 10:41:52 $
   $Author: straten $ */

#ifndef __TimeSeries_h
#define __TimeSeries_h

#include <memory>

#include "Error.h"
#include "environ.h"

#include "dsp/DataSeries.h"

namespace dsp {
  
  class TimeSeries : public DataSeries {

  public:

    //! Automatically delete arrays on resize(0)
    static bool auto_delete;

    //! Null constructor
    TimeSeries ();

    //! Copy constructor
    TimeSeries(const TimeSeries& ts);

    //! Clone operator
    virtual TimeSeries* clone() const;

    //! Returns a null-instantiation (calls new)
    virtual TimeSeries* null_clone() const;

    //! Swaps the two TimeSeries's.  Returns '*this'
    virtual TimeSeries& swap_data(TimeSeries& ts);

    //! Destructor
    virtual ~TimeSeries();

    //! Set this equal to copy
    virtual TimeSeries& operator = (const TimeSeries& copy);

    //! Same as operator= but takes a pointer
    virtual void copy(const TimeSeries* ts)
    { operator=( *ts ); }

    //! Add each value in data to this
    virtual TimeSeries& operator += (const TimeSeries& data);

    //! Multiple each value by this scalar
    virtual TimeSeries& operator *= (float mult);

    //! Copy the configuration of another TimeSeries instance (not the data)
    //! This doesn't copy nchan, npol or ndim if data is being preserved
    virtual void copy_configuration (const Observation* copy);

    //! Copy the data of another TimeSeries instance
    virtual void copy_data (const TimeSeries* data, 
			    uint64 idat_start = 0, uint64 ndat = 0);

    //! Disable the set_nbit method of the Observation base class
    virtual void set_nbit (unsigned);

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (uint64 nsamples);

    //! Use the supplied array to store nsamples time samples.
    //! Always deletes existing data
    virtual void resize( uint64 nsamples, uint64 bytes_supplied, unsigned char* buffer);

    //! Equivalent to resize(0) but instead of deleting data, returns the pointer for reuse elsewhere
    virtual void zero_resize(unsigned char*& _buffer, uint64& nbytes);

    //! For nchan=1, npol=1 data this uses the data in 'buffer'
    TimeSeries& use_data(float* _buffer, uint64 _ndat);
    
    //! Return pointer to the specified data block
    float* get_datptr (unsigned ichan=0, unsigned ipol=0);

    //! Return pointer to the specified data block
    const float* get_datptr (unsigned ichan=0, unsigned ipol=0) const;

    //! Offset the base pointer by offset time samples
    virtual void seek (int64 offset);

    //! Append the given TimeSeries to the end of 'this'
    virtual uint64 append (const TimeSeries*);

    //! Only append the one chan/pol pair
    virtual uint64 append (const TimeSeries*, unsigned ichan, unsigned ipol);

    //! Copy data from given TimeSeries in front of the current position
    void prepend (const dsp::TimeSeries*, uint64 pre_ndat = 0);

    //! Return the sample offset from the start of the data source
    int64 get_input_sample () const { return input_sample; }

    //! Set all values to zero
    virtual void zero ();

    //! Return the mean of the data for the specified channel and poln
    double mean (unsigned ichan, unsigned ipol);

    //! Check that each floating point value is roughly as expected
    virtual void check (float min=-10.0, float max=10.0);

    //! Delete the current data buffer and attach to this one
    virtual void attach (std::auto_ptr<float> _data);

    //! Call this when you do not want to transfer ownership of the array
    virtual void attach (float* _data);

    //! Called by Transformation::operation() to ensure that saved data
    //! stays saved and is not wiped over.
    //! Variable is reset to false after call to transformation()
    void set_preserve_seeked_data(bool _psd){ preserve_seeked_data = _psd; }
   
    //! Inquire whether the data that has been seeked over top of
    //! will be saved on a resize.
    bool get_preserve_seeked_data(){ return preserve_seeked_data; }

    //! Returns how many samples have been seeked over
    uint64 get_seekage();

    //! Over-rides DataSeries::set_nchan()- this only allows a change if preserve_seeked_data is false
    void set_nchan(unsigned _nchan);
    //! Over-rides DataSeries::set_npol()- this only allows a change if preserve_seeked_data is false
    void set_npol(unsigned _npol);
    //! Over-rides DataSeries::set_ndim()- this only allows a change if preserve_seeked_data is false
    void set_ndim(unsigned _ndim);

  protected:

    //! Returns a uchar pointer to the first piece of data
    virtual unsigned char* get_data();
    //! Returns a uchar pointer to the first piece of data
    virtual const unsigned char* get_data() const;

    //! Called by append()
    void append_checks(uint64& ncontain,uint64& ncopy,
		       const TimeSeries* little);
    
    //! Pointer into buffer, offset to the first time sample requested by user
    float* data;

    //! Change the amount of memory reserved at the start of the buffer
    void change_reserve (int64 change) const;

    //! Get the amount of memory reserved at the start of the buffer
    uint64 get_reserve () const { return reserve_ndat; }

    friend class InputBuffering;
    friend class Unpacker;

  private:

    /*! Used by OutputBuffering to ensure that data is saved during resize */
    bool preserve_seeked_data;

    //! Reserve space for this many timesamples preceding the base address
    uint64 reserve_ndat;

    //! Number of floats reserved
    uint64 reserve_nfloat;

    //! Sample offset from start of source
    /*! Set by Unpacker class and used by multithreaded InputBuffering */
    int64 input_sample;

    //! Called by constructor to initialise variables
    void init ();


  };

  class TimeSeriesPtr{
    public:
    
    TimeSeries* ptr;
   
    TimeSeries& operator * () const;
    TimeSeries* operator -> () const; 
        
    TimeSeriesPtr& operator=(const TimeSeriesPtr& tsp);

    TimeSeriesPtr(const TimeSeriesPtr& tsp);
    TimeSeriesPtr(TimeSeries* _ptr);
    TimeSeriesPtr();

    bool operator < (const TimeSeriesPtr& tsp) const;

    ~TimeSeriesPtr();
  };

  // This class just stores a pointer into a particular channel/polarisation pair's data
  class ChannelPtr{
  public :
    TimeSeries* ts;
    unsigned ichan;
    unsigned ipol;
    double centre_frequency;
    
    void init(TimeSeries* _ts,unsigned _ichan, unsigned _ipol);

    ChannelPtr& operator=(const ChannelPtr& c);

    ChannelPtr();
    ChannelPtr(TimeSeries* _ts,unsigned _ichan, unsigned _ipol);
    ChannelPtr(TimeSeriesPtr _ts,unsigned _ichan, unsigned _ipol);
    ChannelPtr(const ChannelPtr& c);
    ~ChannelPtr();
    
    float*& get_ptr();
    const float* get_const_ptr() const;
    bool has_ptr() const;
    void set_ptr(float* _ptr){ ptr = _ptr; }

    bool operator < (const ChannelPtr& c) const;

    float& operator[](unsigned index);

    float get_centre_frequency();

  private:
    float* ptr;
  };
  
}
    
bool operator==(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2);
bool operator!=(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2);

#endif
