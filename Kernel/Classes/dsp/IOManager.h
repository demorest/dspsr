//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/IOManager.h,v $
   $Revision: 1.9 $
   $Date: 2003/08/21 09:47:10 $
   $Author: wvanstra $ */


#ifndef __IOManager_h
#define __IOManager_h

#include "dsp/Input.h"
#include "Error.h"

namespace dsp {

  class BitSeries;
  class Unpacker;
  class TimeSeries;

  //! Convenience interface to different file formats
  /*! This class defines a common interface to backend-specific file
    formats by constructing the appropriate header-loading and
    data-unpacking routines.  */
  class IOManager : public Input {

  public:
    
    //! Constructor
    IOManager ();
    
    //! Destructor
    virtual ~IOManager ();
    
    //! Prepare the appropriate Input and Unpacker
    virtual void open (const char* id);

    //! Prepare the appropriate Input and Unpacker
    void open (const string& id) { open (id.c_str()); }

    //! Return pointer to the appropriate Input
    Input* get_input () const { return input; }
    
    //! Set the Input operator (should not normally need to be used)
    void set_input (Input* input, bool set_params = false);

     //! Return pointer to the appropriate Unpacker
    Unpacker* get_unpacker () const { return unpacker; }

    //! Set the Unpacker (should not normally need to be used)
    void set_unpacker (Unpacker* unpacker);

    //! Set the TimeSeries into which data will be loaded and unpacked
    virtual void set_output (TimeSeries* output);

    //! Set the BitSeries into which data will be loaded
    // (should not normally need to be used)
    virtual void set_output (BitSeries* output);

    //! End of data
    virtual bool eod();
    
    //! Load and convert the next block of data
    virtual void load (TimeSeries* data);

    //! Seek to the specified time sample
    virtual void seek (int64 offset, int whence = 0);
    
    //! Get the number of time samples to load on each call to load_data
    virtual uint64 get_block_size () const { return block_size; }
    //! Set the number of time samples to load on each call to load_data
    virtual void set_block_size (uint64 _size);
    
    //! Return the number of time samples by which consecutive blocks overlap
    virtual uint64 get_overlap () const { return overlap; }
    //! Set the number of time samples by which consecutive blocks overlap
    virtual void set_overlap (uint64 _overlap);
    
  protected:

    //! Define abstract method of the Input base class
    void load_data (BitSeries* data);

    //! Load the BitSeries and/or TimeSeries specified with set_output
    virtual void operation ();

    //! Number of time samples to load on each load_block
    uint64 block_size;

    //! Number of time samples by which data blocks overlap
    uint64 overlap;

    //! Appropriate Input subclass
    Reference::To<Input> input;

    //! Appropriate Unpacker subclass
    Reference::To<Unpacker> unpacker;

    //! The container in which the TimeSeries data is unpacked
    Reference::To<TimeSeries> data;

  private:
    void init();
    
  };

}

#endif // !defined(__IOManager_h)
