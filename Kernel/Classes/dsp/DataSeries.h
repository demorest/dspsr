//-*-C++-*-

#ifndef __DataSeries_h
#define __DataSeries_h

#include "dsp/Observation.h"

//! Container of dimension/time-major order floating point data.
  /* The DataSeries class contains n-bit data arranged as a
     function of frequency, polarization, time, and dimension, ie.

     f0p0t0d0...f0p0t0dD,f0p0t1d0...f0p0t1dD...f0p0tTd0...f0p0tTdD, 
     f0p1t0d0...f0p1t0dD,f0p1t1d0...f0p1t1dD...f0p1tTd0...f0p1tTdD, 
     ...
     f0pPt0d0...f0pPt0dD,f0pPt1d0...f0pPt1dD...f0pPtTd0...f0pPtTdD, 
     f1p0t0d0...f1p0t0dD,f1p0t1d0...f1p0t1dD...f1p0tTd0...f1p0tTdD, 
     ...
     fFpPt0d0...fFpPt0dD,fFpPt1d0...fFpPt1dD...fFpPtTd0...fFpPtTdD
  */

namespace dsp {
  
  class DataSeries : public Observation {
    
  public:
    
    //! Counts number of DataSeries's in existence
    static int instantiation_count;
    //! Stores the cumulative amount of memory used
    static int64 memory_used;
    
    //! Null constructor
    DataSeries ();

    //! Copy constructor
    DataSeries(const DataSeries& ds);

    //! Called by constructor to initialise variables
    virtual void init();

    //! Cloner (calls new)
    virtual DataSeries* clone() = 0;

    //! Returns a null-instantiation (calls new)
    virtual DataSeries* null_clone() = 0;

    //! Swaps the two DataSeries's.  Returns '*this'
    virtual DataSeries& swap_data(DataSeries& ds);

    //! Destructor
    virtual ~DataSeries();

    //! Set this equal to copy
    virtual DataSeries& operator = (const DataSeries& copy);

    //! Enforces that ndat*ndim must be an integer number of bytes
    virtual void set_ndat(uint64 _ndat);

    //! Enforces that ndat*ndim must be an integer number of bytes
    virtual void set_ndim(uint64 _ndim);

    //! Allocate the space required to store nsamples time samples.
    //! Note that the space used in each chan/pol group must be an integer
    //! number of bytes.
    virtual void resize (uint64 nsamples);

    //! Return pointer to the specified block of time samples
    virtual unsigned char* get_udatptr (unsigned ichan=0,unsigned ipol=0);

    //! Return pointer to the specified block of time samples
    virtual const unsigned char* get_udatptr (unsigned ichan=0,unsigned ipol=0) const;

    //! Check to see if subsize==ndat
    virtual bool full(){ return subsize_samples()==get_ndat(); }

    //! Set ndat to subsize- the most it can be
    virtual void set_maximal_ndat(){ set_ndat( subsize_samples() ); }

    //! Inquire the stride (in bytes) between floats representing the same
    //! timesample but in different chan/pol groups.  This is called by CoherentFBWriter 
    virtual uint64 get_subsize(){ return subsize; }

  protected:
    
    //! Returns a uchar pointer to the first piece of data
    virtual unsigned char* get_data();
    //! Returns a uchar pointer to the first piece of data
    virtual const unsigned char* const_get_data() const;

    //! Returns the number of samples in a data sub-division
    //! Note that this assumes that ndat*ndim corresponds to an integer number of bytes
    //! resize() and set_ndat() should enforce this
    uint64 subsize_samples();

    //! The data buffer
    unsigned char* buffer;

    //! The size of the data buffer (in bytes)
    uint64 size;

    //! The number of BYTES in a data sub-division
    uint64 subsize;

  };

}  

#endif
