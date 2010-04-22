//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DataSeries_h
#define __DataSeries_h

#include "dsp/Observation.h"


namespace dsp {

  class Memory;

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
  class DataSeries : public Observation {
    
  public:

    //! Counts number of DataSeries's in existence
    static int instantiation_count;
    //! Stores the cumulative amount of memory used
    static int64_t memory_used;

    //! Null constructor
    DataSeries ();

    //! Copy constructor
    DataSeries(const DataSeries& ds);

    //! Called by constructor to initialise variables
    void initi();

    //! Cloner (calls new)
    virtual DataSeries* clone() const = 0;

    //! Returns a null-instantiation (calls new)
    virtual DataSeries* null_clone() const = 0;

    //! Swaps the two DataSeries's.  Returns '*this'
    virtual DataSeries& swap_data(DataSeries& ds);

    //! Destructor
    virtual ~DataSeries();

    //! Set this equal to copy
    virtual DataSeries& operator = (const DataSeries& copy);

    //! Same as operator= but takes a pointer
    virtual void copy(const DataSeries* ds)
    { operator=( *ds ); }

    //! Enforces that ndat*ndim must be an integer number of bytes
    virtual void set_ndat(uint64_t _ndat);

    //! Enforces that ndat*ndim must be an integer number of bytes
    virtual void set_ndim(uint64_t _ndim);

    //! Allocate the space required to store nsamples time samples.
    //! Note that the space used in each chan/pol group must be an integer
    //! number of bytes.
    virtual void resize (uint64_t nsamples);

    virtual void resize (uint64_t nsamples, unsigned char*& old_buffer);

    //! Reshape the buffer to match the current attributes
    void reshape ();

    //! Reshape the buffer to match the specified attributes
    void reshape (unsigned npol, unsigned ndim);

    //! Set all values to zero
    virtual void zero ();

    //! Return pointer to the specified block of time samples
    virtual unsigned char* get_udatptr (unsigned ichan=0,unsigned ipol=0);

    //! Return pointer to the specified block of time samples
    virtual const unsigned char* get_udatptr (unsigned ichan=0,unsigned ipol=0) const;

    //! Check to see if any more samples can be added
    virtual bool full(){ return maximum_ndat()==get_ndat(); }

    //! Set ndat so to the most it can be with current offset of data from base pointer
    virtual void set_maximal_ndat(){ set_ndat( maximum_ndat() ); }

    //! Inquire the stride (in bytes) between floats representing the same
    //! timesample but in different chan/pol groups.  This is called by CoherentFBWriter 
    virtual uint64_t get_subsize(){ return subsize; }

    //! Returns the maximum ndat possible with current offset of data from base pointer
    virtual uint64_t maximum_ndat() const;

    //! Returns the number of samples that have been seeked over
    virtual int64_t get_samps_offset() const;

    //! Checks that ndat is not too big for size and subsize
    virtual void check_sanity() const;

    //! Match the internal memory layout of another DataSeries
    virtual void internal_match (const DataSeries*);

    //! Return the internal memory base address
    unsigned char* internal_get_buffer() { return buffer; }
    const unsigned char* internal_get_buffer() const { return buffer; }

    //! Return the internal memory size
    uint64_t internal_get_size() const { return size; }

    //! Return the internal memory sub-division size
    uint64_t internal_get_subsize() const { return subsize; }

    //! Set the memory manager
    virtual void set_memory (Memory*);
    const Memory* get_memory () const;

  protected:
    
    //! Returns a uchar pointer to the first piece of data
    virtual unsigned char* get_data();
    //! Returns a uchar pointer to the first piece of data
    virtual const unsigned char* get_data() const;

    //! The data buffer
    unsigned char* buffer;

    //! The size of the data buffer (in bytes)
    uint64_t size;

    //! The number of BYTES in a data sub-division
    uint64_t subsize;

    //! The memory manager
    Reference::To<Memory> memory;

  };

}  

#endif
