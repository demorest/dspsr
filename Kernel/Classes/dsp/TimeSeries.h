//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TimeSeries.h,v $
   $Revision: 1.46 $
   $Date: 2008/10/14 18:53:28 $
   $Author: straten $ */

#ifndef __TimeSeries_h
#define __TimeSeries_h

#include <memory>

#include "Error.h"
#include "environ.h"

#include "dsp/DataSeries.h"

namespace dsp {

  //! Arrays of consecutive samples for each polarization and frequency channel
  class TimeSeries : public DataSeries
  {
  public:

    //! Order of the dimensions
    enum Order {

      //! Frequency, Polarization, Time (default before 3 October 2008)
      OrderFPT,

      //! Time, Frequency, Polarization (better for many things)
      OrderTFP

    };

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

    //! Get the order
    Order get_order () const;

    //! Set the order
    void set_order (Order order);

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

    //! Decrease the array lengths without changing the base pointers
    virtual void decrease_ndat (uint64 new_ndat);

    //! For nchan=1, npol=1 data this uses the data in 'buffer'
    TimeSeries& use_data(float* _buffer, uint64 _ndat);
    
    //! Return pointer to the specified data block
    float* get_datptr (unsigned ichan=0, unsigned ipol=0);

    //! Return pointer to the specified data block
    const float* get_datptr (unsigned ichan=0, unsigned ipol=0) const;

    //! Return pointer to the specified data block
    float* get_dattfp ();

    //! Return pointer to the specified data block
    const float* get_dattfp () const;

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

    void finite_check () const;

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

    //! Order of the dimensions
    Order order;

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
  
}

#endif

