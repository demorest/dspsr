//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Operation.h,v $
   $Revision: 1.15 $
   $Date: 2003/03/07 00:00:14 $
   $Author: pulsar $ */

#ifndef __Operation_h
#define __Operation_h

#include <string>

#include "RealTimer.h"
#include "Reference.h"

namespace dsp {
  
  //! Defines the interface by which operations are performed on data
  /*! This pure virtual base class defines the manner in which various
    digital signal processing routines are performed on baseband data */

  class Operation : public Reference::Able {

  public:

    //! Global flag enables stopwatch to record the time spent operating
    static bool record_time;

    //! Global verbosity flag
    static bool verbose;

    //! All sub-classes must specify a unique name
    Operation (const char* name);

    //! Virtual destructor
    virtual ~Operation ();

    //! Call this method to operate on data
    //! This does time the operation if record_time=true, but doesn't automatically print out the result
    virtual void operate ();

    //! Return the unique name of this operation
    string get_name() { return name; }

    //! Return the total time spent on this Operation in seconds
    double get_total_time () const;

    //! Get the time spent in the last invocation of operate()
    double get_elapsed_time() const;

  protected:

    //! Perform operation on data.  Defined by sub-classes
    virtual void operation () = 0;

    //! Operation name
    string name;

    //! Return pointer to memory resource shared by operations
    static float* float_workingspace (size_t nfloats)
    { return (float*) workingspace (nfloats * sizeof(float)); }
    
    //! Return pointer to memory resource shared by operations
    static double* double_workingspace (size_t ndoubles)
    { return (double*) workingspace (ndoubles * sizeof(double)); }
    
    //! Return pointer to memory resource shared by operations
    static void* workingspace (size_t nbytes);

    //! Stop watch records the amount of time spent performing this operation
    RealTimer optime;

  };
  
}

#endif
