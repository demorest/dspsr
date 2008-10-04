//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Operation.h,v $
   $Revision: 1.41 $
   $Date: 2008/10/04 09:56:19 $
   $Author: straten $ */

#ifndef __Operation_h
#define __Operation_h

#include "dsp/dsp.h"

#include "RealTimer.h"
#include "Reference.h"
#include "environ.h"

#include <string>
#include <vector>
#include <iostream>

namespace dsp {
  
  class Scratch;

  //! Defines the interface by which operations are performed on data
  /*! This pure virtual base class defines the manner in which various
    digital signal processing routines are performed on baseband data */

  class Operation : public Reference::Able {

  public:

    //! Global flag enables stopwatch to record the time spent operating
    static bool record_time;

    //! Global flag enables report of time spent in operation on descruction
    static bool report_time;

    //! Global verbosity flag
    static bool verbose;

    //! Operations can set this to non-zero in operation() if they fail
    static int operation_status;

    //! Operations should perform internal consistency checks
    static bool check_state;

    //! If necessary, operations should buffer their input to prevent data loss
    static bool preserve_data;

    //! Counts how many Operation instantiations there have been
    //! Used for setting the unique instantiation ID
    static int instantiation_count;

    //! All sub-classes must specify a unique name
    Operation (const char* name);

    //! Copy constructor
    Operation (const Operation&);

    //! Virtual destructor
    virtual ~Operation ();

    //! Call this method to operate on data
    //! Returns false on failure
    virtual bool operate ();

    //! Prepare for data operations
    /*! This method enables optimizations by some derived classes */
    virtual void prepare ();

    //! Combine results with another operation
    /*! This method enables results from multiple threads to be combined */
    virtual void combine (const Operation*);

    //! Report operation statistics
    virtual void report () const;

    //! Return the unique name of this operation
    std::string get_name() const { return name; }

    //! Return the total time spent on this Operation in seconds
    double get_total_time () const;

    //! Get the time spent in the last invocation of operate()
    double get_elapsed_time() const;

    //! Return the total number of timesample weights encountered
    virtual uint64 get_total_weights () const;

    //! Return the number of invalid timesample weights encountered
    virtual uint64 get_discarded_weights () const;

    //! Reset the count of invalid timesample weights encountered
    virtual void reset_weights_counters ();

    //! Inquire the unique instantiation id
    int get_id(){ return id; }

    //! Set the scratch space
    virtual void set_scratch (Scratch*);

    //! Set verbosity ostream
    virtual void set_ostream (std::ostream& os) const;

  protected:

    //! Shared scratch space, if needed
    Scratch* scratch;

    //! Return false if the operation doesn't have enough data to proceed
    virtual bool can_operate();

    //! Perform operation on data.  Defined by derived classes
    virtual void operation () = 0;

    //! Set the name!
    virtual void set_name (const std::string& _name){ name = _name; }

    //! Operation name
    std::string name;

    //! Number of time sample weights encountered that are flagged invalid
    uint64 discarded_weights;

    //! Total number of time sample weights encountered
    uint64 total_weights;

    //! Returns the index in the 'timers' array of a particular timer
    int timers_index (const std::string& op_name);

    //! Stop watch records the amount of time spent performing this operation
    RealTimer optime;

    //! Unique instantiation id
    int id;

    //! Set true when preparation optimizations are completed
    bool prepared;

    //! Stream to which verbose messages are sent
    mutable std::ostream cerr;

  };
  
}

#endif
