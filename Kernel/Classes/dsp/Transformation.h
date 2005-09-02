//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Transformation.h,v $
   $Revision: 1.35 $
   $Date: 2005/09/02 07:59:05 $
   $Author: hknight $ */

#ifndef __baseband_dsp_Transformation_h
#define __baseband_dsp_Transformation_h

#include <string>
#include <iostream>
#include <typeinfo>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "Error.h"

namespace dsp {
  class TransformationBase;

  template <class Out>
  class HasOutput;

  template <class In>
  class HasInput;

  template <class In, class Out>
  class Transformation;
}

#include "dsp/Operation.h"
#include "dsp/Observation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BufferingPolicy.h"
#include "dsp/InputBuffering.h"
#include "dsp/OutputBuffering.h"

namespace dsp {

  //! Base class of all Transformation classes
  /*! This interface is highly dangerous and should never be used */
  class TransformationBase : public Operation {
  public:
    TransformationBase (const char* name) : Operation (name) {}
    virtual ~TransformationBase ();

    typedef enum { no_buffering_policy, input_buffering_policy, output_buffering_policy } DefaultBufferingPolicy;

    static DefaultBufferingPolicy default_buffering_policy;

  protected:
    //! Add friend classes only as absolutely necessary
    friend class Simultaneous;
    friend class ProcessingStep;

    virtual void vset_input (void* _input) = 0;
    virtual void vset_output (void* _output) = 0;
    virtual void* vget_input () = 0;
    virtual void* vget_output () = 0;
    virtual string get_input_typestring () = 0;
    virtual string get_output_typestring () = 0;

  };

  template <class In>
  class HasInput {

  public:

    //! Virtual destructor required
    virtual ~HasInput ();

    //! Set the container from which input data will be read
    virtual void set_input (In* _input) { input = _input; }

    //! Return pointer to the container from which input data will be read
    In* get_input () const { return input; }
 
    //! Returns true if input is set
    bool has_input() const { return input.ptr(); }

  protected:

    //! Container from which input data will be read
    Reference::To <In> input;

  };


  template <class Out>
  class HasOutput {

  public:

    //! Virtual destructor required
    virtual ~HasOutput ();

    //! Set the container into which output data will be written
    virtual void set_output (Out* _output) { output = _output; }

    //! Return pointer to the container into which output data will be written
    Out* get_output () const { return output; }

    //! Returns true if output is set
    bool has_output() const { return output.ptr(); }

  protected:

    //! Container into which output data will be written
    Reference::To <Out> output;

  };


  //! All Transformations must define their behaviour
  typedef enum { inplace, outofplace, anyplace } Behaviour;

  //! Defines the interface by which Transformations are performed on data
  /*! This pure virtual template base class defines the manner in
    which data container classes are connected to various digital
    signal processing operations. */
  template <class In, class Out>
  class Transformation : public TransformationBase,
			 public HasInput<In>,
			 public HasOutput<Out>
  {

  public:

    //! All sub-classes must specify name and capacity for inplace operation
    Transformation (const char* _name, Behaviour _type,
		    bool _time_conserved=false);

    //! Destructor
    virtual ~Transformation ();

    //! Set the container from which input data will be read
    void set_input (In* input);

    //! Set the container into which output data will be written
    void set_output (Out* output);

    //! Return the Transformation type
    Behaviour get_type() { return type; }

    //! Set the policy for buffering input and/or output data
    virtual void set_buffering_policy (BufferingPolicy* policy)
    { buffering_policy = policy; }

    BufferingPolicy* get_buffering_policy () const;

    //! Convenience method
    Reference::To<OutputBuffering<In> > get_outputbuffering();
 
    //! Returns true if buffering_policy is set
    bool has_buffering_policy() const { return buffering_policy.ptr(); }

    //! Reset minimum_samps_can_process
    void reset_min_samps()
    { minimum_samps_can_process = -1; }

    //! Inquire whether the class conserves time
    bool get_time_conserved(){ return time_conserved; }

    //! to add a dspExtension history object to the output
    virtual void add_history(){ }

  protected:

    //! The buffering policy in place (if any)
    Reference::To<BufferingPolicy> buffering_policy;

    //! Return false if the input doesn't have enough data to proceed
    virtual bool can_operate();

    //! Define the Operation pure virtual method
    virtual void operation ();

    //! Declare that sub-classes must define a transformation method
    virtual void transformation () = 0;

    //! If input doesn't have this many samples, operate() returns false
    int64 minimum_samps_can_process;

    /** @name TransformationBase interface
     *  These kludgey methods should never be used by anyone.
     */
    //@{

    virtual string get_input_typestring()
    { return typeid(this->input.ptr()).name(); }

    virtual string get_output_typestring()
    { return typeid(this->output.ptr()).name(); }

    virtual void vset_input(void* _input)
    { this->set_input( (In*)_input ); }

    virtual void vset_output(void* _output)
    { this->set_output( (Out*)_output ); }

    virtual void* vget_input()
    { return const_cast<void*>((const void*)this->get_input()); }

    virtual void* vget_output()
    { return this->get_output(); }

    //@}

  private:

    //! Makes sure input & output are okay before calling transformation()
    void checks();

    //! Behaviour of Transformation
    Behaviour type;

    //! If output is a container, its ndat is rounded off to divide this number
    uint64 rounding;

    //! Returns true if the Transformation definitely conserves time
    /*! (i.e. it conserves time if the number of seconds in the output
      corresponds to the number of seconds in the input processed).
      Acceleration classes don't conserve time. This must be set in
      the constructor to be true if it is true- some constructors may
      conserve time but may not yet have had their constructors change
      to reflect this [false] */
    bool time_conserved;


  };
  
}

//! All sub-classes must specify name and capacity for inplace operation
template<class In, class Out>
dsp::Transformation<In,Out>::Transformation(const char* _name, 
					    Behaviour _type,
					    bool _time_conserved)
  : TransformationBase (_name)
{
  if( Operation::verbose )
    fprintf(stderr,"In Transformation constructor for '%s'\n",_name);

  type = _type;
  reset_min_samps();
  time_conserved = _time_conserved;

  if( default_buffering_policy == output_buffering_policy ){
    Transformation<In,TimeSeries>* tr = dynamic_cast<Transformation<In,TimeSeries>*>(this);
    if( tr && _type != inplace )
      buffering_policy = new OutputBuffering<In>(tr);
  }
  else if( default_buffering_policy == input_buffering_policy ){
    Transformation<TimeSeries,TimeSeries>* tr = dynamic_cast<Transformation<TimeSeries,TimeSeries>*>(this);
    if( tr && _type != inplace )
      buffering_policy = new InputBuffering(tr);
  }
}

//! Return false if the input doesn't have enough data to proceed
template<class In, class Out>
bool dsp::Transformation<In,Out>::can_operate()
{
  /* This is neither type-safe nor run-time safe - WvS
     if( type==inplace && !has_input() && has_output() )
       input = (In*)output.get();
  */

  if (!this->has_input())
    return false;

  if (minimum_samps_can_process < 0)
    return true;

  if (int64(this->get_input()->get_ndat()) >= minimum_samps_can_process)
    return true;

  if( verbose )
    cerr << "dsp::Transformation<In,Out> (" << get_name() << ")"
      " has input of " << this->get_input()->get_ndat() << " samples."
      "  Minimum is " << minimum_samps_can_process << endl;

  return false;
}

//! Makes sure input & output are okay before calling transformation()
template <class In, class Out>
void dsp::Transformation<In, Out>::checks(){
  // If inplace is true, then the input and output should be of the same type....
  if( type==inplace && !this->input.ptr() && this->output.ptr() )
    this->input = (In*)this->output.get();
  if( type==inplace && !this->output.ptr() && this->input.ptr() )
    this->output = (Out*)this->input.get();
  
  if (!this->input)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "no input");

  if (this->input->get_ndat() < 1)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "empty input- input=%p input->ndat="UI64,
		 this->input.get(),this->input->get_ndat());

  string reason;
  if (check_state && !this->input->state_is_valid (reason))
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "invalid input state: " + reason);

  if ( type!=inplace && !this->output)
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "no output");
}

//! Define the Operation pure virtual method
template <class In, class Out>
void dsp::Transformation<In, Out>::operation ()
{
  checks();

  if (buffering_policy)
    buffering_policy -> pre_transformation ();

  transformation ();

  if (buffering_policy)
    buffering_policy -> post_transformation ();

  string reason;
  if (check_state && type!=inplace && !this->output->state_is_valid (reason))
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "invalid output state: " + reason);

  add_history();  
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_input (In* _input)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_input ("<<_input<<")"<<endl;

  this->input = _input;

  if ( type == outofplace && this->input && this->output
       && (const void*)this->input == (const void*)this->output )
    throw Error (InvalidState, "dsp::Transformation["+name+"]::set_input",
		 "input must != output");

  if( type==inplace )
    this->output = (Out*)_input;
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_output (Out* _output)
{
  if (verbose)
    cerr << "dsp::Transformation["+name+"]::set_output ("<<_output<<")"<<endl;

  if (type == inplace && this->input 
      && (const void*)this->input != (const void*)_output )
    throw Error (InvalidState, "dsp::Transformation["+name+"]::set_output",
		 "inplace transformation input must equal output");
  
  if ( type == outofplace && this->input && this->output 
       && (const void*)this->input.get() == (const void*)_output )
    throw Error (InvalidState, "dsp::Transformation["+name+"]::set_output",
		 "output must != input");

  this->output = _output;

  if( type == inplace && !this->has_input() )
    this->input = (In*)_output;

}

//! Convenience method
template <class In, class Out>
Reference::To<dsp::OutputBuffering<In> >
dsp::Transformation<In,Out>::get_outputbuffering(){
  if( Operation::verbose )
    fprintf(stderr,"dsp::Transformation<In,Out>::get_outputbuffering() for '%s' with %p\n",
	    get_name().c_str(),buffering_policy.ptr());

  if( !has_buffering_policy() )
    buffering_policy = new OutputBuffering<In>(this);
  else if( buffering_policy->get_name() != "OutputBuffering" )
    throw Error(InvalidState,"dsp::Transformation::get_outputbuffering()",
		"Buffering policy was not of correct OutputBuffering type! (this = '%s') (It was '%s')",
		get_name().c_str(),buffering_policy->get_name().c_str());

  OutputBuffering<In>* ret = dynamic_cast<OutputBuffering<In>*>(buffering_policy.get());

  Reference::To<dsp::OutputBuffering<In> > retb = ret;

  return retb;
}

template <class In, class Out>
dsp::BufferingPolicy* dsp::Transformation<In,Out>::get_buffering_policy () const {
  return buffering_policy;
}

template <class In, class Out>
dsp::Transformation<In,Out>::~Transformation(){
  if( Operation::verbose ){
    fprintf(stderr,"Transformation (%s) destructor entered input=%p output=%p\n",
	    get_name().c_str(),input.ptr(),output.ptr());
    if( input.ptr() )
      fprintf(stderr,"Transformation (%s) destructor input has %d refs\n",
	      get_name().c_str(), input->get_reference_count());
    if( output.ptr() )
      fprintf(stderr,"Transformation (%s) destructor output has %d refs\n",
	      get_name().c_str(), output->get_reference_count());
  }
}

template <class Out>
dsp::HasOutput<Out>::~HasOutput(){ }

template <class In>
dsp::HasInput<In>::~HasInput(){ }




#endif
