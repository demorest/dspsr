//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Input.h

#ifndef __Input_h
#define __Input_h

#include "dsp/Operation.h"
#include "dsp/Observation.h"

#include "MJD.h"

class ThreadContext;

namespace dsp {

  class BitSeries;

  //! Pure virtual base class of all objects that can load BitSeries data
  /*! 
    This class defines the common interface as well as some basic
    functionality relating to sources of BitSeries data.
  */

  class Input : public Operation {

    friend class IOManager;

  public:
    
    //! Constructor
    Input (const char* name);
    
    //! Destructor
    virtual ~Input ();

    //! The origin of the data may be re-implemented by wrappers like MultiFile
    virtual const Input* get_origin () const { return this; }

    //! Copies the behaviour and information attributes (not data)
    virtual void copy (const Input* input);

    //! Prepare the output with the attributes of the data source
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Reserve the maximum amount of space required in the given container
    void reserve (BitSeries*);

    //! End of data
    virtual bool eod() = 0;
    
    //! Load BitSeries data
    /*! Only this load method is guaranteed to be thread safe */
    void load (BitSeries* data);

    //! Set the BitSeries to which data will be loaded
    virtual void set_output (BitSeries* data);

    //! Retrieve a pointer to the output.
    virtual BitSeries* get_output();

    //! Return true if output is set
    virtual bool has_output () const;

    //! Seek to the specified time sample
    virtual void seek (int64_t offset, int whence = 0);

    //! Return the first time sample to be read on the next call to operate
    uint64_t tell () const { return load_sample; }

    //! Seek to a sample close to the specified MJD
    virtual void seek (MJD mjd);
    
    //! Return the number of time samples to load on each load_block
    virtual uint64_t get_block_size () const { return block_size; }
    //! Set the number of time samples to load on each load_block
    virtual void set_block_size (uint64_t _size);
    
    //! Return the number of time samples by which consecutive blocks overlap
    virtual uint64_t get_overlap () const { return overlap; }
    //! Set the number of time samples by which consecutive blocks overlap
    virtual void set_overlap (uint64_t _overlap) { overlap = _overlap; }

    //! Convenience function for returning block_size-overlap
    virtual uint64_t get_stride () const 
    { return get_block_size()-get_overlap(); }

    //! Return the total number of time samples available
    virtual uint64_t get_total_samples () const { return get_info()->get_ndat(); }

    //! Set the total number of time samples available
    /*! Generally useful for debugging */
    void set_total_samples (uint64_t s) { get_info()->set_ndat(s); }

    //! Get the information about the data source
    operator const Observation* () const { return get_info(); }

    //! Get the information about the data source
    virtual Observation* get_info () { return info; }

    //! Get the information about the data source
    virtual const Observation* get_info () const { return info; }

    //! Get the next time sample to be loaded
    uint64_t get_load_sample () const { return load_sample; }

    //! Get the number of samples to be loaded
    uint64_t get_load_size () const { return load_size; }

    //! Get the time sample resolution of the data source
    unsigned get_resolution () const { return resolution; }

    //! Convenience method used to seek in units of seconds
    void seek_seconds (double seconds, int whence = 0);

    //! Convenience method used to report the offset in seconds
    double tell_seconds () const;

    //! Set the start of observation offset in units of seconds
    void set_start_seconds (double seconds);

    //! Convenience method used to set the number of seconds
    void set_total_seconds (double seconds);

    //! In multi-threaded programs, a mutual exclusion and a condition
    void set_context (ThreadContext* context);

    //! Input derived types may specify a prefix to be added to output files
    virtual std::string get_prefix () const;

  protected:

    //! Set the 'end_of_data' flag in dsp::Seekable
    virtual void set_eod (bool) = 0;

    //! The BitSeries to which data will be loaded on next call to operate
    Reference::To<BitSeries> output;

    //! Load the next block of time samples into BitSeries
    /*! Implementations of this method must read get_load_size time
      samples, beginning with get_load_sample */
    virtual void load_data (BitSeries* data) = 0;

    //! Load data into the BitSeries specified with set_output
    virtual void operation ();

    //! Mark the output BitSeries with sequence informatin
    virtual void mark_output ();

    //! Information about the data source (passed on to BitSeries in load)
    Reference::To<Observation> info;

    //! Time sample resolution of the data source
    /*! Derived classes must define the smallest number of time
      samples that can be loaded at one time.  For instance, when reading
      two-bit sampled data, four time samples may be packed into one
      byte.  In this case, resolution == 4. */
    unsigned resolution;
    
    //! The ndat of the BitSeries last loaded
    //! Used by Seekable::recycle_data() and set by load()
    uint64_t last_load_ndat;

    //! If not "" then the source of the output gets changed to this after loading [""]
    std::string real_source;

  private:

    //! Requested number of time samples to be read during load
    /*! The number of times samples actually read during the call to
      load_data is given by get_load_size.  This may be greater than
      block_size, owing to the resolution of the data source.
      Attributes of both the BitSeries and TimeSeries classes keep
      these details transparent at the top level. */
    uint64_t block_size;
        
    //! Requested number of time samples by which data blocks overlap
    /*! The number of times samples by which data blocks overlap
      actually depends on the resolution of the data source.
      Attributes of both the BitSeries and TimeSeries classes keep
      these details transparent at the top level. */
    uint64_t overlap;
    
    //! Offset from load_sample to the time sample requested by the user
    /*! Owing to the resolution of the data source, the first sample
      read during load_data may precede the time sample requested by
      the user.  Attributes of both the BitSeries and TimeSeries
      classes keep these details transparent at the top level. */
    unsigned resolution_offset;

    //! First time sample to be read on the next call to load_data
    uint64_t load_sample;

    //! Number of time samples to be read on the next call to load_data
    uint64_t load_size;

    //! Offset into data stream set by set_start_seconds
    uint64_t start_offset;

    //! Ensures that load size is properly set
    void set_load_size ();

    //! Thread coordination used in Input::load method
    ThreadContext* context;
  };

}

#endif // !defined(__Input_h)
