//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/IOManager.h,v $
   $Revision: 1.6 $
   $Date: 2002/11/06 06:30:42 $
   $Author: hknight $ */


#ifndef __IOManager_h
#define __IOManager_h

#include "Error.h"

#include "dsp/Input.h"
#include "dsp/Unpacker.h"
#include "dsp/Chronoseries.h"
#include "dsp/Timeseries.h"

namespace dsp {

  class Operation;

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
    void set_input (Input* input);

    //! Because IOManager doesn't write to a Chronoseries, it needs this:
    virtual void set_output(Chronoseries* _output)
    { throw Error(FailedCall,"dsp::IOManager::set_output()",
		  "For an IOManager, use set_final_output, not set_output!"); }

    //! This is the one to call:
    virtual void set_final_output(Timeseries* _output);

    //! Return pointer to the appropriate Unpacker
    Unpacker* get_unpacker () const { return unpacker; }

    //! Set the Unpacker (should not normally need to be used)
    void set_unpacker (Unpacker* unpacker);

    // not sure about the rest of these

    //! End of data
    virtual bool eod();
    
    //! Load and convert the next block of data
    virtual void load (Timeseries* data);

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
    
    //! Get the number of samples used to estimate undigitized power
    virtual int get_nsample () const { return nsample; }

    //! Set the number of samples used to estimate undigitized power
    virtual void set_nsample (int _nsample);

    //! Set the container into which input (raw) data will be read
    virtual void set_raw (Chronoseries* raw);

  protected:

    //! Yet another Haydon kludge
    Reference::To<Timeseries> output;
    
    //! Null-define abstract method of the Input base class
    void load_data (Chronoseries* data);

    //! Number of time samples to load on each load_block
    uint64 block_size;

    //! Number of time samples by which data blocks overlap
    uint64 overlap;

    //! Number of samples used to estimate undigitized power
    int nsample;

    //! Appropriate Input subclass
    Reference::To<Input> input;

    //! Appropriate Unpacker subclass
    Reference::To<Unpacker> unpacker;

    //! The container in which the input (raw, unpacked) bitstream is stored
    Reference::To<Chronoseries> raw;

  private:
    void init();
    
  };

}

#endif // !defined(__IOManager_h)
