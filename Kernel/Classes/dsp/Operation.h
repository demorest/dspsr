//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Operation.h,v $
   $Revision: 1.12 $
   $Date: 2002/11/06 06:30:42 $
   $Author: hknight $ */

#ifndef __Operation_h
#define __Operation_h

#include <string>

#include "RealTimer.h"
#include "Reference.h"

namespace dsp {
  
  class Basicseries;

  //! Defines the interface by which operations are performed on Basicseries
  /*! This pure virtual base class defines the manner in which various
    digital signal processing routines are performed on the baseband
    data contained in a Basicseries object */

  class Operation : public Reference::Able {

  public:

    //! All operations must define their behaviour
    enum Behaviour { inplace, outofplace, anyplace };

    //! Global flag enables stopwatch to record the time spent operating
    static bool record_time;

    //! Global verbosity flag
    static bool verbose;

    //! All sub-classes must specify name and capacity for inplace operation
    Operation (const char* name, Behaviour type);

    //! Virtual destructor
    virtual ~Operation ();

    //! Call this method to operate on input Basicseries
    virtual void operate ();

    //! Return a string that describes the operation
    //virtual const string descriptor () const = 0;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor) = 0;

    //! Set the container from which input data will be read
    //! Over-ride this to check input is of right type (use dynamic_cast)
    virtual void set_input (const Basicseries* input);

    //! Set the container into which output data will be written
    //! Over-ride this to check output is of right type (use dynamic_cast)
    virtual void set_output (Basicseries* output);

    //! Return pointer to the container from which input data will be read
    virtual const Basicseries* get_input () const;

    //! Return pointer to the container into which output data will be written
    virtual Basicseries* get_output () const;

    Behaviour get_type() { return type; }

    string get_name() { return name; }

    //! Return the total time spent on this Operation in seconds
    double get_total_time () const;

  protected:

    //! check the input is of right type- called by set_input()
    virtual void check_input() = 0;

    //! check the output is of right type- called by set_output()
    virtual void check_output() = 0;

    //! Perform operation on data.  Defined by sub-classes
    virtual void operation () = 0;

    //! Operation name
    string name;

    //! Container from which input data will be read
    Reference::To <const Basicseries> input;

    //! Container into which output data will be written
    Reference::To <Basicseries> output;

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
