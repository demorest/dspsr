//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Operation.h,v $
   $Revision: 1.33 $
   $Date: 2007/05/25 21:38:10 $
   $Author: straten $ */

#ifndef __Operation_h
#define __Operation_h

#include "dsp/dsp.h"
#include "dsp/Time.h"

#include "RealTimer.h"
#include "Reference.h"
#include "environ.h"

#include <string>
#include <vector>

namespace dsp {
  
  class Scratch;
  class TimeKeeper;

  //! Defines the interface by which operations are performed on data
  /*! This pure virtual base class defines the manner in which various
    digital signal processing routines are performed on baseband data */

  class Operation : public Reference::Able {

    friend class TimeKeeper;

  public:

    //! If this is set to true then dsp::Transformation nukes prepends with test values
    static bool debug;

    //! Global flag enables stopwatch to record the time spent operating
    static bool record_time;

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

    //! Virtual destructor
    virtual ~Operation ();

    //! Call this method to operate on data
    //! Returns false on failure
    virtual bool operate ();

    //! Return the unique name of this operation
    std::string get_name() { return name; }

    //! Return the total time spent on this Operation in seconds
    double get_total_time () const;

    //! Get the time spent in the last invocation of operate()
    double get_elapsed_time() const;

    //! Return the number of invalid timesample weights encountered
    virtual uint64 get_discarded_weights () const;

    //! Reset the count of invalid timesample weights encountered
    virtual void reset_discarded_weights ();

    //! Only ever called by TimeKeeper class
    static void set_timekeeper(TimeKeeper* _timekeeper);
    static void unset_timekeeper();

    //! Inquire the unique instantiation id
    int get_id(){ return id; }

    //! Pointer to the timekeeper
    static TimeKeeper* timekeeper;

  protected:

    //! Shared scratch space, if needed
    Reference::To<Scratch> scratch;

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

    //! Returns the index in the 'timers' array of a particular timer
    int timers_index (const std::string& op_name);

    //! Called by TimeKeeper to print out times etc.
    Time get_operation_time();
    std::vector<Time> get_extra_times();

    //! Stop watch records the amount of time spent performing this operation
    OperationTimer optime;

    //! More stop watches for recording miscellaneous timings
    std::vector<OperationTimer> timers;

    //! Unique instantiation id
    int id;
  };
  
}

#endif
