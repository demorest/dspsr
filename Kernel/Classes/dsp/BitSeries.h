//-*-C++-*-

/*

This class is not pure virtual, but it probably will be in future.

*/

#ifndef __BitSeries_h
#define __BitSeries_h

#include "dsp/Observation.h"

namespace dsp {
  
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

    //! Set the number of bits per value
    void set_nbit (unsigned _nbit) { nbit = _nbit; }
    //! Return the number of polarizations
    unsigned get_nbit () const { return nbit; }

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (uint64 nsamples);
    
    virtual bool combinable (const BitSeries& bs) const;

    //! Set this equal to data
    virtual BitSeries& operator = (const BitSeries& data);

    //! Set the Observation component of this equal to obs
    BitSeries& operator = (const Observation& obs)
    { Observation::operator= (obs); return *this; }

    //! Return pointer to the raw data buffer
    virtual unsigned char* get_rawptr () { return data; }

    //! Return pointer to the raw data buffer
    virtual const unsigned char* get_rawptr () const { return data; }


    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual unsigned char* get_datptr (uint64 sample);

    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual const unsigned char* get_datptr (uint64 sample) const;

    //! Append little onto the end of this
    virtual void append (const BitSeries* little);

    //! Return the sample offset from the start of the data source
    int64 get_input_sample () const { return input_sample; }

    //! Return the size in bytes of nsamples time samples
    uint64 nbytes (uint64 nsamples) const
      { return (nsamples*nbit*npol*nchan*ndim)/8; }

    //! Return the size in bytes of ndat time samples
    uint64 nbytes () const
      { return nbytes (ndat); }

    //! Return the size in bytes of one time sample
    float nbyte () const
      { return float(nbit*npol*nchan*ndim) / 8.0; }

    //! Return the number of samples in nbytes bytes
    uint64 nsamples (uint64 nbytes) const
      { return (nbytes * 8)/(nbit*npol*nchan*ndim); }

  protected:
    //! Number of bits per value
    unsigned nbit;

    //! The data buffer
    unsigned char *data;

    //! The size of the data buffer (not necessarily ndat)
    uint64 size;

    //! Sample offset from start of source; attribute used by Input class
    int64 input_sample;

  };
  
}

#endif
