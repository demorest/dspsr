//-*-C++-*-

#ifndef __BitSeries_h
#define __BitSeries_h

#include <memory>

#include "Reference.h"

#include "dsp/MiniPlan.h"
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

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (int64 nsamples);
    
    //! Set this equal to data
    virtual BitSeries& operator = (const BitSeries& data);

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
    uint64 get_request_ndat () const { return request_ndat; }

    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual unsigned char* get_datptr (uint64 sample);

    //! Return pointer to the specified time slice (ie ch0,pol0,dim0)
    virtual const unsigned char* get_datptr (uint64 sample) const;

    //! Append little onto the end of this
    virtual void append (const BitSeries* little);

    //! Return the sample offset from the start of the data source
    int64 get_input_sample (Input* input = 0) const;

    //! Delete the current data buffer and attach to this one
    //! This is dangerous as it ASSUMES new data buffer has been pre-allocated and is big enough.  Beware of segmentation faults when using this routine.
    //! Also do not try to delete the old memory once you have called this- the BitSeries::data member now owns it.
    virtual void attach(auto_ptr<unsigned char> _data);

    //! Call this when you want the array to still be owned by it's owner
    virtual void attach(unsigned char* _data);

    //! Release control of the data buffer- resizes to zero
    virtual unsigned char* release(uint64& _size);
    //! For use by MiniSeries to share the data buffer for unpacking
    void share(unsigned char*& _buffer,uint64& _size) const;

    //! Retrieve a reference to the miniplan 
    Reference::To<MiniPlan> get_miniplan() const { return miniplan; }

    //! Set the miniplan- usually called by MiniFile loader
    void set_miniplan(Reference::To<MiniPlan> _miniplan){ miniplan = _miniplan; }

    //! Returns true if the miniplan is set
    bool has_miniplan() const { return miniplan.ptr(); }

  protected:
    //! The data buffer
    unsigned char* data;

    //! The size (in bytes) of the allocated data buffer
    //! Note that more space may have been allocated than used by the 'ndat' samples
    int64 size;

    //! Sample offset from start of source; attribute used by Input class
    int64 input_sample;

    //! The Input instance to last set input_sample
    Input* input;

    //! Offset (owing to resolution) to the requested time sample
    unsigned request_offset;

    //! Number of time samples requested
    uint64 request_ndat;

    //! The MiniPlan, if one is needed to expand into a TimeSeries
    Reference::To<MiniPlan> miniplan;

  };
  
}

#endif
