//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Transformation.h,v $
   $Revision: 1.2 $
   $Date: 2002/11/10 01:56:48 $
   $Author: wvanstra $ */

#ifndef __Transformation_h
#define __Transformation_h

#include "dsp/Operation.h"
#include "Error.h"

namespace dsp {

  //! Defines the interface by which Transformations are performed on data
  /*! This pure virtual template base class defines the manner in
    which various digital signal processing routines are performed. */
  template <class In, class Out>
  class Transformation : public Operation {

  public:

    //! All operations must define their behaviour
    enum Behaviour { inplace, outofplace, anyplace };

    //! All sub-classes must specify name and capacity for inplace operation
    Transformation (const char* _name, Behaviour _type) : Operation (_name)
    { type = _type; }

    //! Virtual destructor
    virtual ~Transformation () { }

    //! Set the container from which input data will be read
    //! Over-ride this to check input is of right type (use dynamic_cast)
    virtual void set_input (In* input);

    //! Set the container into which output data will be written
    //! Over-ride this to check output is of right type (use dynamic_cast)
    virtual void set_output (Out* output);

    //! Return pointer to the container from which input data will be read
    virtual In* get_input () const { return input; }
 
    //! Return pointer to the container into which output data will be written
    virtual Out* get_output () const { return output; }

    //! Return the Transformation type
    Behaviour get_type() { return type; }

  protected:

    //! Define the Operation pure virtual method
    virtual void operation ();

    //! Declare that sub-classes must define a transformation method
    virtual void transformation () = 0;

    //! Container from which input data will be read
    Reference::To <In> input;

    //! Container into which output data will be written
    Reference::To <Out> output;

  private:

    //! Behaviour of Transformation
    Behaviour type;

  };
  
}

template <class In, class Out>
void dsp::Transformation<In, Out>::operation ()
{
  if (verbose)
    cerr << "Transformation["+name+"]::operate" << endl;

  if (!input)
    throw Error (InvalidState, "Transformation["+name+"]::operate",
		 "no input");

  if (input->get_ndat() < 1)
    throw Error (InvalidState, "Transformation["+name+"]::operate",
		 "empty input");

  string reason;
  if (!input->state_is_valid (reason))
    throw Error (InvalidState, "Transformation["+name+"]::operate",
		 "invalid input state: " + reason);

  if (!inplace && !output)
    throw Error (InvalidState, "Transformation["+name+"]::operate",
		 "no output");

  //! call the pure virtual method defined by sub-classes
  transformation ();

  if (!inplace && !output->state_is_valid (reason))
    throw Error (InvalidState, "Transformation["+name+"]::operate",
		 "invalid output state: " + reason);
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_input (In* _input)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_input ("<<_input<<")"<<endl;

  input = _input;

  if ( type == outofplace && input && output
       && (const void*)input == (const void*)output )
    throw Error (InvalidState, "Transformation["+name+"]::set_input",
		 "input must != output");
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_output (Out* _output)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_output ("<<_output<<")"<<endl;

  if (type == inplace)
    throw Error (InvalidState, "Transformation["+name+"]::set_output",
		 "inplace transformation has only input");

  output = _output;

  if ( type == outofplace && input && output 
       && (const void*)input == (const void*)output )
    throw Error (InvalidState, "Transformation["+name+"]::set_output",
		 "output must != input");
}


#endif
