//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Input.h,v $
   $Revision: 1.1 $
   $Date: 2002/09/25 06:12:23 $
   $Author: wvanstra $ */


#ifndef __Input_h
#define __Input_h

#include "Observation.h"
#include "RealTimer.h"

namespace dsp {

  class Timeseries;

  //! Pure virtual base class of all objects that can load Timeseries data
  /*! 
    This class defines the common interface as well as some basic
    functionality relating to sources of baseband data.
  */
  class Input : public Reference::Able {

  public:
    
    //! Verbosity flag
    static bool verbose;
    
    //! Constructor
    Input () { init(); }
    
    //! Destructor
    virtual ~Input () { }
    
    //! End of data
    virtual bool eod() = 0;
    
    //! Load Timeseries data
    virtual void load (Timeseries* data);

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
    
    //! Set the information about the data source
    virtual void set_info (const Observation& _info) { info = _info; }

    //! Get the information about the data source
    virtual const Observation& get_info () const { return info; }

  protected:

    //! Load next block of data into Timeseries
    virtual void load_data (Timeseries* data) = 0;

    //! Information about the data source (passed on to Timeseries in load)
    Observation info;

    //! Number of time samples to load on each load_block
    uint64 block_size;
    //! Number of time samples by which data blocks overlap
    uint64 overlap;
    //! First time sample to be read on the next call to load_block
    uint64 next_sample;
    
    //! Conserve access to resources by re-using data already in Timeseries
    uint64 recycle_data (Timeseries* data);
    
  private:
    //! Initialize all attributes with null values
    void init();
    
    //! Stop watch records the amount of time spent in load method
    RealTimer load_time;

  };

}

#endif // !defined(__Input_h)
