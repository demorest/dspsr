//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Transformation.h,v $
   $Revision: 1.13 $
   $Date: 2004/03/30 04:58:40 $
   $Author: hknight $ */

#ifndef __Transformation_h
#define __Transformation_h

#include <string>
#include <iostream>

#include <stdlib.h>

#include "environ.h"
#include "Error.h"

#include "dsp/Operation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! All operations must define their behaviour
  typedef enum { inplace, outofplace, anyplace } Behaviour;

  //! Defines the interface by which Transformations are performed on data
  /*! This pure virtual template base class defines the manner in
    which various digital signal processing routines are performed. */
  template <class In, class Out>
  class Transformation : public Operation {

  public:

    //! All sub-classes must specify name and capacity for inplace operation
    Transformation (const char* _name, Behaviour _type) : Operation (_name)
    { type = _type; free_scratch_space = swap_buffers = false; }

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

    //! Returns true if input is set
    virtual bool has_input(){ return input.ptr(); }

    //! Returns true if output is set
    virtual bool has_output(){ return output.ptr(); }

    //! Return the Transformation type
    Behaviour get_type() { return type; }

    //! Setting this determines whether you want to swap 'input' and 'output' before returning
    //! You might set this to true when you have a class that must be outofplace, but you want
    //!   your output to go into the same TimeSeries as your input.
    void set_swap_buffers(bool _swap_buffers){ swap_buffers = _swap_buffers; }

    //! Inquire whether the 'input' and 'output' will be swapped before returning
    bool get_swap_buffers(){ return swap_buffers; }

    //! Setting this determines whether you want to delete the unused output buffer
    //! Use this when you have 'swap_buffers' set to true, and you don't want the TimeSeries that was used as output
    void set_free_scratch_space(bool _free_scratch_space){ free_scratch_space = _free_scratch_space; }

    //! Inquire whether you want to delete the unused output buffer
    bool get_free_scratch_space(){ return free_scratch_space; }

  protected:

    //! Define the Operation pure virtual method
    virtual void operation ();

    //! Declare that sub-classes must define a transformation method
    virtual void transformation () = 0;

    //! Container from which input data will be read
    Reference::To <In> input;

    //! Container into which output data will be written
    Reference::To <Out> output;

    //! Swap 'input' and 'output' before returning (simulates an inplace operation but can be faster) (Only for TimeSeries's)
    //! You might set this to true when you have a class that must be outofplace, but you want
    //!   your output to go into the same TimeSeries as your input.
    bool swap_buffers;

    //! If 'swap_buffers' is true, and 'free_scratch_space' is true, then the 'output' is resized to zero to free up memory (Only for TimeSeries's)
    //! Use this when you have 'swap_buffers' set to true, and you don't want the TimeSeries that was used as output
    bool free_scratch_space;

  private:

    //! Behaviour of Transformation
    Behaviour type;

  };
  
}

template <class In, class Out>
void dsp::Transformation<In, Out>::operation ()
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::operate" << endl;

  // If inplace is true, then the input and output should be of the same type....
  if( type==inplace && !input.ptr() && output.ptr() )
    input = (In*)output.get();
  if( type==inplace && !output.ptr() && input.ptr() )
    output = (Out*)input.get();

  if (!input)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "no input");

  if (input->get_ndat() < 1)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "empty input- input=%p input->ndat="UI64,
		 input.get(),input->get_ndat());

  string reason;
  if (!input->state_is_valid (reason))
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "invalid input state: " + reason);

  if ( type!=inplace && !output)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "no output");

  //! call the pure virtual method defined by sub-classes
  transformation ();

  if ( type!=inplace && !output->state_is_valid (reason))
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "invalid output state: " + reason);

  if( swap_buffers ){
    // Perhaps a better idea would be each class having a 'name' attribute?
    if( sizeof(In)==sizeof(TimeSeries) && sizeof(Out)==sizeof(TimeSeries) ){
      TimeSeries* in = (TimeSeries*)input.ptr();
      TimeSeries* out = (TimeSeries*)output.ptr();

      in->swap_data( *out );
      if( free_scratch_space )
	out->resize(0);
    }
  }

}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_input (In* _input)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_input ("<<_input<<")"<<endl;

  input = _input;

  if ( type == outofplace && input && output
       && (const void*)input == (const void*)output )
    throw Error (InvalidState, "dsp::Transformation["+name+"]::set_input",
		 "input must != output");

  if( type==inplace )
    output = (Out*)_input;
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_output (Out* _output)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_output ("<<_output<<")"<<endl;

  if (type == inplace && input.ptr() && (const void*)input.get()!=(const void*)_output )
    throw Error(InvalidState, "dsp::Transformation["+name+"]::set_output",
		 "inplace transformation input must equal output");
  
  output = _output;

  if ( type == outofplace && input && output 
       && (const void*)input.get() == (const void*)output.get() ){
    Error er(InvalidState, "dsp::Transformation["+name+"]::set_output",
		 "output must != input");
    cerr << er << endl;
    exit(-1);
  }

  if( type == inplace && !input.ptr() )
    input = (In*)_output;

}


#endif
