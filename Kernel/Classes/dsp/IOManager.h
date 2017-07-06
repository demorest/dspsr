//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/IOManager.h


#ifndef __IOManager_h
#define __IOManager_h

#include "dsp/Operation.h"

namespace dsp {

  class Observation;
  class Input;
  class BitSeries;
  class Unpacker;
  class TimeSeries;

  //! Convenience interface to different file formats
  /*! This class defines a common interface to backend-specific file
    formats by constructing the appropriate header-loading and
    data-unpacking routines.  */
  class IOManager : public Operation {

  public:
    
    //! Constructor
    IOManager ();
    
    //! Destructor
    virtual ~IOManager ();
    
    //! Prepare the input and unpacker
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Add any input and unpacker extensions
    void add_extensions (Extensions*);

    //! Combine the input and unpacker
    void combine (const Operation*);

    //! Reset the input and unpacker
    void reset ();

    //! Report operation statistics
    void report () const;

    //! Prepare the appropriate Input and Unpacker
    virtual void open (const std::string& id);

    //! Get the information about the data source
    const Observation* get_info () const;
    Observation* get_info ();

    //! Return pointer to the appropriate Input
    const Input* get_input () const;
    Input* get_input ();
 
    //! Set the Input operator (should not normally need to be used)
    void set_input (Input* input);

     //! Return pointer to the appropriate Unpacker
    const Unpacker* get_unpacker () const;
    Unpacker* get_unpacker ();

    //! Set the Unpacker (should not normally need to be used)
    void set_unpacker (Unpacker* unpacker);

    //! Set the TimeSeries into which data will be loaded and unpacked
    virtual void set_output (TimeSeries* output);

    //! Set the BitSeries into which data will be loaded
    // (should not normally need to be used)
    virtual void set_output (BitSeries* output);

    //! Set the maximum RAM usage constraint in set_block_size
    void set_maximum_RAM (uint64_t);
    //! Set the minimum RAM usage constraint in set_block_size
    void set_minimum_RAM (uint64_t);
    //! Set the number of copies of data constraint in set_block_size
    void set_copies (unsigned);

    //! Set custom post load operation
    void set_post_load_operation (Operation * op);

    //! Set the resolution of the filterbank
    virtual void set_filterbank_resolution (unsigned res);

    //! Set the block size according to above constraints
    virtual uint64_t set_block_size (uint64_t minimum_samples);

    virtual void set_overlap (uint64_t overlap);

    //! Load and convert the next block of data
    virtual void load (TimeSeries* data);

    //! Set the scratch space
    virtual void set_scratch (Scratch*);

    //! Set verbosity ostream
    virtual void set_cerr (std::ostream& os) const;


  protected:

    //! Load the BitSeries and/or TimeSeries specified with set_output
    virtual void operation ();

    //! Appropriate Input subclass
    Reference::To<Input> input;

    //! Appropriate Unpacker subclass
    Reference::To<Unpacker> unpacker;

    //! The container in which the TimeSeries data is unpacked
    Reference::To<BitSeries> output;

    //! The container in which the TimeSeries data is unpacked
    Reference::To<TimeSeries> data;

    //! Optional operation performed between load and unpack
    Reference::To<Operation> post_load_operation;

    uint64_t maximum_RAM;
    uint64_t minimum_RAM;
    unsigned copies;
    unsigned filterbank_resolution;
  };

}

#endif // !defined(__IOManager_h)
