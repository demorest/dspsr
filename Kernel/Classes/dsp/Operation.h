//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Operation.h,v $
   $Revision: 1.2 $
   $Date: 2002/06/30 07:34:33 $
   $Author: pulsar $ */

#ifndef __Operation_h
#define __Operation_h

#include <string>
#include "RealTimer.h"

namespace dsp {
  
  class Timeseries;

  //! Defines the interface by which operations are performed on Timeseries
  /*! This pure virtual base class defines the manner in which various
    digital signal processing routines are performed on the baseband
    data contained in a Timeseries object */

  class Operation {

  public:

    //! All operations must define their behaviour
    enum Behaviour { inplace, outofplace, anyplace };

    //! Global flag tells all operations to record the time spent operating
    static bool record_time;

    //! Global verbosity flag
    static bool verbose;

    //! All sub-classes must specify name and capacity for inplace operation
    Operation (const char* name, Behaviour type);

    //! Virtual destructor
    virtual ~Operation ();

    //! Call this method to operate on input Timeseries
    virtual void operate ();

    //! Return a string that describes the operation
    virtual const string descriptor () const = 0;

    //! Initialize from a descriptor string as output by above
    virtual void initialize (const string& descriptor) = 0;

    //! Set the container from which input data will be read
    virtual void set_input (const Timeseries* input);

    //! Set the container into which output data will be written
    virtual void set_output (Timeseries* output);

    //! Return pointer to the container from which input data will be read
    virtual const Timeseries* get_input () const;

    //! Return pointer to the container into which output data will be written
    virtual Timeseries* get_output () const;

  protected:

    //! Perform operation on data.  Defined by sub-classes
    virtual void operation () = 0;

    //! Reset the Timeseries::loader_sample attribute
    virtual void reset_loader_sample ();

    //! Operation name
    string name;

    //! Container from which input data will be read
    const Timeseries* input;

    //! Container into which output data will be written
    Timeseries* output;

    //! Return pointer to memory resource shared by operations
    static float* float_workingspace (size_t nfloats)
    { return (float*) workingspace (nfloats * sizeof(float)); }
    
    //! Return pointer to memory resource shared by operations
    static double* double_workingspace (size_t ndoubles)
    { return (double*) workingspace (ndoubles * sizeof(double)); }
    
    //! Return pointer to memory resource shared by operations
    static void* workingspace (size_t nbytes);

  private:
    //! Stop watch records the amount of time spent performing this operation
    RealTimer optime;

    //! Behaviour of operation
    Behaviour type;

  };
  
}

#endif
