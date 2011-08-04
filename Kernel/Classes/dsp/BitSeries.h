//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __BitSeries_h
#define __BitSeries_h

#include <memory>

#include "dsp/Observation.h"
#include "dsp/Memory.h"

namespace dsp {
  
  class Input;

  //! A container for storing digitized (generally not floating point) data   
  /*! A BitSeries is a container that can be used to store data from
  some device.  In general, devices are considered to produce data as
  a function of time.  Therefore, time will generally be the most
  slowly changing dimension in a BitSeries, though this is not
  necessarily true at the bit level in some file formats.  The
  BitSeries may also be used to store digitized data before it is
  written to disk.  */

  class BitSeries : public Observation {

    friend class Input;

  public:
    //! Null constructor
    BitSeries ();

    //! Destructor
    virtual ~BitSeries ();

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (int64_t nsamples);
    
    //! Set this equal to data
    virtual BitSeries& operator = (const BitSeries& data);

    //! Same as operator= but takes a pointer
    virtual void copy(const BitSeries* bs)
    { operator=( *bs ); }

    //! Set the Observation component of this equal to obs
    BitSeries& operator = (const Observation& obs)
    { Observation::operator= (obs); return *this; }

    //! Return pointer to the raw data buffer
    virtual unsigned char* get_rawptr () { return data; }

    //! Return pointer to the raw data buffer
    virtual const unsigned char* get_rawptr () const { return data; }

    //! Offset (owing to resolution) to the requested time sample
    unsigned get_request_offset () const { return request_offset; }

    //! Number of time samples requested
    uint64_t get_request_ndat () const { return request_ndat; }

    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual unsigned char* get_datptr (uint64_t sample = 0);

    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual const unsigned char* get_datptr (uint64_t sample = 0) const;

    //! Copy the data of another BitSeries instance
    virtual void copy_data (const BitSeries* data, 
			    uint64_t idat_start = 0, uint64_t ndat = 0);

    uint64_t get_size () const { return data_size; }

    //! Append little onto the end of this
    virtual void append (const BitSeries* little);

    //! Set the sample offset from start of the data source
    void set_input_sample (int64_t sample) { input_sample = sample; }

    //! Return the sample offset from the start of the data source
    int64_t get_input_sample (Input* input = 0) const;

    const Input* get_loader() const { return input; }

    void set_memory (Memory*);

    //! Match the internal memory layout of another BitSeries
    void internal_match (const BitSeries*);

    //! Copy the configuration of another TimeSeries instance (not the data)
    void copy_configuration (const Observation* copy);

  protected:

    friend class Unpacker;

    //! The data buffer
    unsigned char* data;

    //! The size (in bytes) of the allocated data buffer
    /*! Note that more space may have been allocated than indicated by
      the ndat attribute */
    int64_t data_size;

    //! Sample offset from start of source; attribute used by Input class
    int64_t input_sample;

    //! Offset (owing to resolution) to the requested time sample
    unsigned request_offset;

    //! Number of time samples requested
    uint64_t request_ndat;

    //! The Input instance to last set input_sample
    Input* input;

    //! The memory manager
    Reference::To<Memory> memory;

  };
  
}

#endif
