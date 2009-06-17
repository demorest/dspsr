//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Transformation.h,v $
   $Revision: 1.51 $
   $Date: 2009/06/17 10:16:53 $
   $Author: straten $ */

#ifndef __dsp_Transformation_h
#define __dsp_Transformation_h

#include "dsp/Operation.h"
#include "dsp/HasInput.h"
#include "dsp/HasOutput.h"
#include "dsp/BufferingPolicy.h"

#include "Callback.h"
#include "Error.h"

#include <iostream>

namespace dsp {

  //! All Transformations must define their behaviour
  typedef enum { inplace, outofplace, anyplace } Behaviour;

  //! Defines the interface by which Transformations are performed on data
  /*! This template base class defines the manner in which data
    container classes are connected to various digital signal
    processing operations. */
  template <class In, class Out>
  class Transformation : public Operation,
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
    void set_input (const In* input);

    //! Set the container into which output data will be written
    void set_output (Out* output);

    //! Return the Transformation type
    Behaviour get_type() const { return type; }

    //! Set the policy for buffering input and/or output data
    virtual void set_buffering_policy (BufferingPolicy* policy)
    { buffering_policy = policy; }

    //! Returns true if buffering_policy is set
    bool has_buffering_policy() const
    { return buffering_policy; }

    BufferingPolicy* get_buffering_policy () const
    { return buffering_policy; }

    //! Functions called before the transformation takes place
    Callback<Transformation*> pre_transformation;

    //! Functions called after the transformation takes place
    Callback<Transformation*> post_transformation;

    //! Reset minimum_samps_can_process
    void reset_min_samps()
    { minimum_samps_can_process = -1; }

    //! Inquire whether the class conserves time
    bool get_time_conserved() const { return time_conserved; }

    //! String preceding output in verbose mode
    std::string name (const std::string& function) 
    { return "dsp::Tranformation["+Operation::get_name()+"]::" + function; }

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const
    {
      Operation::set_cerr (os);
      if (this->input)
        this->input->set_cerr (os);
      if (this->output)
        this->output->set_cerr (os);
      if (this->buffering_policy)
        this->buffering_policy->set_cerr (os);
    }

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
    int64_t minimum_samps_can_process;

    //! Makes sure input & output are okay before calling transformation()
    virtual void vchecks();

  private:

    //! Behaviour of Transformation
    Behaviour type;

    //! If output is a container, its ndat is rounded off to divide this number
    uint64_t rounding;

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
dsp::Transformation<In,Out>::Transformation (const char* _name, 
					     Behaviour _type,
					     bool _time_conserved)
  : Operation (_name)
{
  if (Operation::verbose)
    cerr << name("ctor") << std::endl;

  type = _type;
  reset_min_samps();
  time_conserved = _time_conserved;
}

//! Return false if the input doesn't have enough data to proceed
template<class In, class Out>
bool dsp::Transformation<In,Out>::can_operate()
{
  if (!this->has_input())
    return false;

  if (minimum_samps_can_process < 0)
    return true;

  if (int64_t(this->get_input()->get_ndat()) >= minimum_samps_can_process)
    return true;

  if (Operation::verbose)
    cerr << name("can_operate") <<
      " input ndat=" << this->get_input()->get_ndat() <<
      " min=" << minimum_samps_can_process << std::endl;

  return false;
}

//! Makes sure input & output are okay before calling transformation()
template <class In, class Out>
void dsp::Transformation<In, Out>::vchecks()
{
  if (type == inplace) {
    if (Operation::verbose)
      cerr << name("vchecks") << " inplace checks" << std::endl;
    // when inplace, In == Out
    if( !this->input && this->output )
      this->input = (In*) this->output.get();
    if( !this->output && this->input )
      this->output = (Out*) this->input.get();
  }

  if (Operation::verbose)
    cerr << name("vchecks") << " input checks" << std::endl;
  
  if (!this->input)
    throw Error (InvalidState, name("vchecks"), "no input");
  
  if (type!=inplace && !this->output)
    throw Error (InvalidState, name("vchecks"), "no output");

  if (Operation::verbose)
    cerr << name("vchecks") << " done" << std::endl;
}

//! Define the Operation pure virtual method
template <class In, class Out>
void dsp::Transformation<In, Out>::operation () try
{
  if (Operation::verbose)
    cerr << name("operation") << " call vchecks" << std::endl;

  vchecks();

  pre_transformation.send (this);

  if (buffering_policy)
  {
    if (Operation::verbose)
      cerr << name("operation") <<
	"\n  calling " + buffering_policy->get_name() + "::pre_transformation"
        " input sample=" << this->get_input()->get_input_sample() << std::endl;

    buffering_policy -> pre_transformation ();
  }

  if (Operation::verbose)
    cerr << name("operation") << " transformation" << std::endl;

  transformation ();

  if (buffering_policy) {
    if (Operation::verbose)
      cerr << name("operation") << " post_transformation" << std::endl;
    buffering_policy -> post_transformation ();
  }

  if (Operation::verbose)
    cerr << name("operation") << " check output" << std::endl;

  post_transformation.send(this);
}
 catch (Error& error) {
   throw error += name("operation");
 }


template <class In, class Out>
void dsp::Transformation<In, Out>::set_input (const In* _input)
{
  if (Operation::verbose)
    cerr << "dsp::Transformation["+this->get_name()+"]::set_input ("<<_input<<")"<<std::endl;

  this->input = _input;

  if ( type == outofplace && this->input && this->output
       && (const void*)this->input == (const void*)this->output )
    throw Error (InvalidState, "dsp::Transformation["+this->get_name()+"]::set_input",
		 "input must != output");

  if( type==inplace )
    this->output = (Out*)_input;
}

template <class In, class Out>
void dsp::Transformation<In, Out>::set_output (Out* _output)
{
  if (Operation::verbose)
    cerr << "dsp::Transformation["+this->get_name()+"]::set_output ("<<_output<<")"<<std::endl;

  if (type == inplace && this->input 
      && (const void*)this->input != (const void*)_output )
    throw Error (InvalidState, "dsp::Transformation["+this->get_name()+"]::set_output",
		 "inplace transformation input must equal output");
  
  if ( type == outofplace && this->input && this->output 
       && (const void*)this->input.get() == (const void*)_output )
    throw Error (InvalidState, "dsp::Transformation["+this->get_name()+"]::set_output",
		 "output must != input");

  this->output = _output;

  if( type == inplace && !this->has_input() )
    this->input = (In*)_output;

}


template <class In, class Out>
dsp::Transformation<In,Out>::~Transformation()
{
  if (Operation::verbose)
    cerr << name("dtor") << std::endl;
}

#endif
