//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Input.h,v $
   $Revision: 1.14 $
   $Date: 2002/12/11 10:13:50 $
   $Author: wvanstra $ */

#ifndef __Input_h
#define __Input_h

#include "dsp/Operation.h"
#include "dsp/Observation.h"
#include "environ.h"

namespace dsp {

  class BitSeries;

  //! Pure virtual base class of all objects that can load BitSeries data
  /*! 
    This class defines the common interface as well as some basic
    functionality relating to sources of BitSeries data.
  */

  class Input : public Operation {

  public:
    
    //! Constructor
    Input (const char* name);
    
    //! Destructor
    virtual ~Input ();
    
    //! End of data
    virtual bool eod() = 0;
    
    //! Load BitSeries data
    virtual void load (BitSeries* data);

    //! Set the BitSeries to which data will be loaded
    virtual void set_output (BitSeries* data);

    //! Seek to the specified time sample
    virtual void seek (int64 offset, int whence = 0);
    
    //! Return the number of time samples to load on each load_block
    virtual uint64 get_block_size () const { return block_size; }
    //! Set the number of time samples to load on each load_block
    virtual void set_block_size (uint64 _size) { block_size = _size; }
    
    //! Return the number of time samples by which consecutive blocks overlap
    virtual uint64 get_overlap () const { return overlap; }
    //! Set the number of time samples by which consecutive blocks overlap
    virtual void set_overlap (uint64 _overlap) { overlap = _overlap; }
    
    //! Change the number of samples the overlap is
    virtual void change_overlap(uint64 _extra_overlap) { overlap += _extra_overlap; }

    //! Return the total number of time samples available
    virtual uint64 get_total_samples () const { return info.get_ndat(); }

    //! Get the information about the data source
    virtual operator const Observation* () const { return &info; }

    //! Get the information about the data source
    virtual const Observation* get_info () const { return &info; }

    //! Get the next time sample
    uint64 get_next_sample () { return next_sample; }

  protected:

    //! The BitSeries to which data will be loaded on next call to operate
    Reference::To<BitSeries> output;

    //! Load next block of data into BitSeries
    virtual void load_data (BitSeries* data) = 0;

    //! Load data into the BitSeries specified with set_output
    virtual void operation ();

    //! Information about the data source (passed on to BitSeries in load)
    Observation info;

    //! Number of time samples to load on each load_block
    uint64 block_size;
        
    //! Number of time samples by which data blocks overlap
    uint64 overlap;
    
    //! First time sample to be read on the next call to load_data
    uint64 next_sample;
    
  };

}

#endif // !defined(__Input_h)
