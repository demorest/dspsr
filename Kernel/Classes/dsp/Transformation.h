//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Transformation.h,v $
   $Revision: 1.19 $
   $Date: 2004/11/01 23:43:43 $
   $Author: hknight $ */

#ifndef __Transformation_h
#define __Transformation_h

#include <string>
#include <iostream>
#include <typeinfo>

#include <math.h>
#include <stdlib.h>

#include "environ.h"
#include "Error.h"

#include "dsp/Operation.h"
#include "dsp/Observation.h"
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
    Transformation (const char* _name, Behaviour _type, bool _time_conserved=false);

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

    //! Reset minimum_samps_can_process
    void reset_min_samps(){ minimum_samps_can_process = -1; }

    //! Save the last_nsamps into the saved_data buffer
    //! DO NOT try and save data for Transformations that accumulate their output- save_data will not work
    void save_data(uint64 last_nsamps); 

    //! Deletes the saved_data buffer
    void delete_saved_data(){ saved_data = 0; }

    //! Returns how many samples were lost in the last call to operation()
    //! Throws an Error if input/output aren't dsp::TimeSeries's
    virtual int64 get_input_samps_lost();

    //! Inquire whether valid data is saved already
    //! DO NOT try and save data for Transformations that accumulate their output- save_data will not work
    bool get_valid_data_is_saved(){ return valid_data_is_saved; }

    //! Set the new rounding factor
    void set_rounding(uint64 _rounding){ rounding = _rounding; }

    //! Inquire the new rounding factor
    uint64 get_rounding(){ return rounding; }

    //! Inquire whether the class conserves time
    bool get_time_conserved(){ return time_conserved; }

    //! Decide whether you want to skip over samples that have been processed already (requires time_conserved==true; TimeSeries->TimeSeries)
    void set_process_samps_once(bool _process_samps_once);

    //! Inquire whether you are going to skip over samples that have been processed already (requires time_conserved==true; Observation->Observation to be meaningful)
    bool get_process_samps_once(){ return process_samps_once; }

  protected:

    //! Return false if the input doesn't have enough data to proceed
    virtual bool can_operate();

    //! Define the Operation pure virtual method
    virtual void operation ();

    //! Declare that sub-classes must define a transformation method
    virtual void transformation () = 0;

    //! Prepends the output with the 'saved_data' buffer after the call to transformation()
    //! Not used now
    void prepend_output(dsp::TimeSeries* ts_out);

    //! Prepends the output with the 'saved_data' buffer before the call to transformation()
    uint64 prepend_data(dsp::TimeSeries* ts_out);

    //! Does all the swap buffer stuff
    void swap_buffer_stuff();

    //! Sets the valid_data_is_saved flag to false after a transformation
    virtual void set_valid_data_is_saved(){ valid_data_is_saved = false; }

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

    //! If >= zero, and input doesn't have this many samples, operate() returns false
    int64 minimum_samps_can_process;

  private:

    //! Prints a string used by Haydon for debugging
    void debug_print(dsp::Observation* ts_in,string str,bool want_print);

    //! Makes sure input & output are okay before calling transformation()
    void checks();

    //! If the input is a dsp::Observation, this returns the input duration
    double get_time_in(dsp::Observation* ts_in);

    //! If the input is a dsp::Observation, this returns the input rate
    double get_rate_in(dsp::Observation* ts_in);

    //! If the input is a dsp::Observation, this returns the input start time
    MJD get_input_start_time(dsp::Observation* ts_in);

    //! Handles the rounding stuff
    void rounding_stuff(dsp::TimeSeries* ts_out);

    //! Called after transformation() to allow the prepended data to be seen by the next Transformation
    void deprepend_data(dsp::TimeSeries* ts_out,int64 samples_prepended,
			double time_in, double rate_in, double time_surplus,
			bool seeked_data_being_saved);

    //! Seeks over any samples that have already been processed
    //! Returns how many samples were seeked over
    int64 seek_over_surplus_samps();

    //! Makes sure the input isn't changed by the seeking over of surplus samples
    void seek_back_over_surplus_samps(int64 surplus_samples);

    //! After transformation() this works out what the MJD of the last sample processed was
    void workout_end_of_processed_data(MJD input_start_time, double time_prepended,
				       double time_surplus);

    //! Behaviour of Transformation
    Behaviour type;

    //! This buffer saves data from last time
    //! DO NOT try and save data for Transformations that accumulate their output- save_data will not work
    Reference::To<Out> saved_data;

    //! Gets set to true whenever valid data is copied over into the 'saved_data' buffer
    bool valid_data_is_saved;

    //! Stores how many samples were lost in the last call to operation()
    int64 input_samps_lost;

    //! If output is a TimeSeries, its ndat is rounded off to divide this number
    uint64 rounding;

    //! Returns true if the Transformation definitely conserves time
    //! (i.e. it conserves time if the number of seconds in the output corresponds to the number of seconds in the input processed)
    //! Acceleration classes don't conserve time
    //! This must be set in the constructor to be true if it is true- some constructors may conserve time but may not yet have had their constructors change to reflect this [false]
    bool time_conserved;

    //! Whether or not any samps that have already been processed will be automatically skipped over.  (Only for Observation->Observation) [false]
    bool process_samps_once;

    //! Stores where the last point that was fully processed in the last input TimeSeries ended
    MJD end_of_processed_data;

  };
  
}

//! All sub-classes must specify name and capacity for inplace operation
template<class In, class Out>
dsp::Transformation<In,Out>::Transformation(const char* _name, Behaviour _type, bool _time_conserved) : Operation (_name)
{
  type = _type;
  free_scratch_space = false;
  swap_buffers = false;
  reset_min_samps();
  input_samps_lost = 0;
  rounding = 0;
  valid_data_is_saved = false; 
  time_conserved = _time_conserved;
  set_process_samps_once( false );
  end_of_processed_data = MJD::zero;
}

//! Return false if the input doesn't have enough data to proceed
template<class In, class Out>
bool dsp::Transformation<In,Out>::can_operate() {
  if( type==inplace && !has_input() && has_output() )
    input = (In*)output.get();
  
  if( has_input() && minimum_samps_can_process >= 0 ){
    if( int64(get_input()->get_ndat()) < minimum_samps_can_process ){
      if( verbose )
	fprintf(stderr,"dsp::Transformation<In,Out> (%s) has input of "I64" samples.  Minimum samps can process is "I64"\n",
		get_name().c_str(),
		int64(get_input()->get_ndat()),
		minimum_samps_can_process);
      return false;
    }
  }

  return true;
}

//! Prints a string used by Haydon for debugging
template <class In, class Out>
void dsp::Transformation<In,Out>::debug_print(dsp::Observation* obs_in,string str,bool want_print){
  if( !want_print || !obs_in )
    return;
  if( verbose ){
    MJD st = MJD("52644.180875615555578");
    fprintf(stderr,"TRANS (%s): At %s.  Input start (%s) is at %f samples past start MJD; ndat="UI64"\n",
	    str.c_str(),
	    get_name().c_str(),
	    obs_in->get_start_time().printdays(15).c_str(),
	    (obs_in->get_start_time()-st).in_seconds()*obs_in->get_rate(),
	    obs_in->get_ndat());
    TimeSeries* ts_in = dynamic_cast<TimeSeries*>(obs_in);
    if( ts_in )
      fprintf(stderr,"TRANS (%s): At %s.  samps_offset="I64"\n",
	      str.c_str(),get_name().c_str(),ts_in->get_samps_offset());
  }
}

//! Makes sure input & output are okay before calling transformation()
template <class In, class Out>
void dsp::Transformation<In, Out>::checks(){
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
}

//! If the input is a dsp::Observation, this returns the input duration
template <class In, class Out>
double dsp::Transformation<In, Out>::get_time_in(dsp::Observation* ts_in){
  if( ts_in )
    return ts_in->get_duration();
  return 0.0;
}

//! If the input is a dsp::Observation, this returns the input rate
template <class In, class Out>
double dsp::Transformation<In, Out>::get_rate_in(dsp::Observation* ts_in){
  if( ts_in )
    return ts_in->get_rate();
  return 0.0;
}

//! If the input is a dsp::Observation, this returns the input start time
template <class In, class Out>
MJD dsp::Transformation<In, Out>::get_input_start_time(dsp::Observation* ts_in){
  if( ts_in )
    return ts_in->get_start_time();
  return MJD::zero;
}

//! Handles the rounding stuff
template <class In, class Out>
void dsp::Transformation<In,Out>::rounding_stuff(dsp::TimeSeries* ts_out){
  if( !ts_out )
    return;
  if( rounding > 0 ){
    uint64 old_ndat = ts_out->get_ndat();
    ts_out->set_ndat( ts_out->get_ndat() - ts_out->get_ndat()%rounding );
    if( verbose )
      fprintf(stderr,"Transformation %s has wiped "UI64" samps\n",
	      get_name().c_str(),
	      old_ndat - ts_out->get_ndat());
  }
}

//! Called after transformation() to allow the prepended data to be seen by the next Transformation
template <class In, class Out>
void dsp::Transformation<In,Out>::deprepend_data(dsp::TimeSeries* ts_out,int64 samples_prepended,double time_in,
						 double rate_in,double time_surplus, bool seeked_data_being_saved){
  if( !ts_out )
    return;
  
  if( samples_prepended > 0 ){
    ts_out->seek( -samples_prepended );

    if( verbose )
      fprintf(stderr,"TRANS deprepend_data (%s) have seeked back "I64" samps offset="I64"\n",
	      get_name().c_str(),samples_prepended,ts_out->get_samps_offset());
    
    if( samples_prepended >= 4 && debug ){
      unsigned twos = 0;
      for( unsigned ichan=0;ichan<ts_out->get_nchan();ichan++){
	for( unsigned ipol=0;ipol<ts_out->get_npol();ipol++){
	  if( fabs(ts_out->get_datptr(ichan,ipol)[2]-2.0)<0.3 )
	    twos++;
	  else
	    fprintf(stderr,"TRANS deprepend_data (%s) ichan=%d ipol=%d sample was not 2- it was %f\n",
		    get_name().c_str(),ichan,ipol,ts_out->get_datptr(ichan,ipol)[2]);
	}
      }
        
      if( ts_out->get_nchan() > 6 )
	fprintf(stderr,"TRANS::deprepend_data() (%s) after seeking backwards "I64" samples have samp [2]=%f [3]=%f twos=%d/%d\n",
		get_name().c_str(), samples_prepended, ts_out->get_datptr(6,0)[2], ts_out->get_datptr(6,0)[3],
		twos,ts_out->get_nchan()*ts_out->get_npol()); 
    }
  }

  if( seeked_data_being_saved )
    ts_out->set_preserve_seeked_data( false );
  
  double time_out = ts_out->get_duration();
  if( verbose )
    fprintf(stderr,"TRANS (%s): time_in=%f time_out=%f time_surplus=%f so input_samps_lost=nint64(%f*%f)\n",
	    get_name().c_str(), time_in, time_out, time_surplus,
	    (time_in-time_surplus)-time_out, rate_in);
  input_samps_lost = nint64( ((time_in-time_surplus)-time_out)*rate_in );
}

//! Seeks over any samples that have already been processed
template <class In, class Out>
int64 dsp::Transformation<In,Out>::seek_over_surplus_samps(){
  dsp::TimeSeries* ts_in = (dsp::TimeSeries*)dynamic_cast<const dsp::TimeSeries*>(get_input());
  dsp::Observation* obs_out = (dsp::Observation*)dynamic_cast<const dsp::Observation*>(get_input());
  
  if( !process_samps_once || !ts_in || !obs_out || end_of_processed_data==MJD::zero || !get_time_conserved() || (void*)get_input()==(void*)get_output() )
    return 0;
  
  double secs_surplus = (end_of_processed_data - ts_in->get_start_time()).in_seconds();
  int64 samps_surplus = nint64(secs_surplus * ts_in->get_rate());
  
  if( samps_surplus < 0 )
    throw Error(InvalidState,"dsp::Transformation::seek_over_surplus_samps()",
		"Your last processing call ended at MJD %s.  This is %f seconds ("I64" samps) before current input starts",
		end_of_processed_data.printdays(15).c_str(),
		fabs(secs_surplus), -samps_surplus);

  if( samps_surplus != 0 ){
    if( verbose ){
      MJD st = MJD("52644.180875615555578");    
      fprintf(stderr,"TRANS::seek_over_surplus_samps() (%s) eopd=%f ts_in->st=%f seeking over %f seconds or "I64" surplus samps\n",
	      get_name().c_str(),
	      (end_of_processed_data-st).in_seconds(),
	      (ts_in->get_start_time()-st).in_seconds(),
	      secs_surplus,samps_surplus);
    }
    ts_in->seek( samps_surplus );
  }

  return samps_surplus;
}

//! After transformation() this works out what the MJD of the last sample processed was
template <class In, class Out>
void dsp::Transformation<In, Out>::workout_end_of_processed_data(MJD input_start_time, double time_prepended,
								 double time_surplus){
  if( !get_time_conserved() )
    return;
  
  dsp::TimeSeries* ts_in = (dsp::TimeSeries*)dynamic_cast<const dsp::TimeSeries*>(get_input());
  dsp::Observation* obs_out = (dsp::Observation*)dynamic_cast<const dsp::Observation*>(get_output());
  if( !ts_in || !obs_out )
    return;

  end_of_processed_data = input_start_time + obs_out->get_duration() - time_prepended + time_surplus;
  if( verbose && get_name()=="Detection" ){
    MJD st = MJD("52644.180875615555578"); 
    fprintf(stderr,"\nTRANS(%s)::workout_end_of_processed_data() has set end_of_processed_data to %f + %f - %f + %f = %f\n\n",
	    get_name().c_str(),
	    (input_start_time-st).in_seconds(),obs_out->get_duration(),
	    time_prepended,time_surplus,
	    (end_of_processed_data-st).in_seconds());
  }
}

//! Makes sure the input isn't changed by the seeking over of surplus samples
template <class In, class Out>
void dsp::Transformation<In, Out>::seek_back_over_surplus_samps(int64 surplus_samples){
  dsp::TimeSeries* ts_in = (dsp::TimeSeries*)dynamic_cast<const dsp::TimeSeries*>(get_input());
  
  if( ts_in )
    ts_in->seek( -surplus_samples );
}

//! Define the Operation pure virtual method
template <class In, class Out>
void dsp::Transformation<In, Out>::operation ()
{
  dsp::Observation* obs_in = (dsp::Observation*)dynamic_cast<const dsp::Observation*>(get_input());

  debug_print(obs_in,"start of operation",verbose);

  checks();

  double time_in = get_time_in(obs_in);
  double rate_in = get_rate_in(obs_in);
  MJD input_start_time = get_input_start_time(obs_in);

  dsp::TimeSeries* ts_out = dynamic_cast<dsp::TimeSeries*>(get_output());

  int64 samples_prepended = 0;

  if( valid_data_is_saved && ts_out ){
    samples_prepended = prepend_data( ts_out );
    time_in += samples_prepended / ts_out->get_rate();
  }

  int64 surplus_samples = seek_over_surplus_samps();

  if( ts_out && ts_out->get_nchan()>6 && samples_prepended>=4 && debug ){
    uint64 data_offset = ts_out->get_samps_offset()*ts_out->get_ndim();
    fprintf(stderr,"TRANS::operation() (%s) before transformation() has floats offset="UI64" ndim=%d [2]=%f (%p) [3]=%f\n",
	    get_name().c_str(), data_offset, ts_out->get_ndim(),
	    (ts_out->get_datptr(6,0)-data_offset)[2],
	    &((ts_out->get_datptr(6,0)-data_offset)[2]),
	    (ts_out->get_datptr(6,0)-data_offset)[3]);
  }

  //debug_print(obs_in,"before transformation",verbose);
  transformation ();
  //debug_print(ts_out,"after transformation",verbose);

  if( ts_out && ts_out->get_nchan()>6 && samples_prepended>=4 && debug ){
    uint64 data_offset = ts_out->get_samps_offset()*ts_out->get_ndim();
    fprintf(stderr,"TRANS::operation() (%s) after transformation() has floats offset="UI64" ndim=%d [2]=%f (%p) [3]=%f\n",
	    get_name().c_str(), data_offset,ts_out->get_ndim(),
	    (ts_out->get_datptr(6,0)-data_offset)[2],
	    &((ts_out->get_datptr(6,0)-data_offset)[2]),
	    (ts_out->get_datptr(6,0)-data_offset)[3]);
  }

  seek_back_over_surplus_samps(surplus_samples);

  rounding_stuff(ts_out);

  deprepend_data(ts_out,samples_prepended,time_in,rate_in,
		 double(surplus_samples)/rate_in,
		 samples_prepended!=0);

  workout_end_of_processed_data(input_start_time,double(samples_prepended)/rate_in,
				double(surplus_samples)/rate_in);
 
  string reason;
  if ( type!=inplace && !output->state_is_valid (reason))
    throw Error (InvalidState, "dsp::Transformation["+name+"]::operate",
		 "invalid output state: " + reason);

  swap_buffer_stuff();

  set_valid_data_is_saved();

  debug_print(ts_out,"end of operation()",verbose);

  
  if( ts_out && ts_out->get_nchan()>6 && samples_prepended >= 4 && debug){
    unsigned twos = 0;
    for( unsigned ichan=0;ichan<ts_out->get_nchan();ichan++){
      for( unsigned ipol=0;ipol<ts_out->get_npol();ipol++){
	if( fabs(ts_out->get_datptr(ichan,ipol)[2]-2.0)<0.3 )
	  twos++;
      }
    }

    fprintf(stderr,"TRANS::operation() (%s) right at end have samp [2]=%f [3]=%f twos=%d/%d\n",
	    get_name().c_str(), ts_out->get_datptr(6,0)[2], ts_out->get_datptr(6,0)[3],
	    twos,ts_out->get_nchan()*ts_out->get_npol()); 
  }
  
}


//! Does all the swap buffer stuff
template <class In, class Out>
void dsp::Transformation<In,Out>::swap_buffer_stuff(){
  if( !swap_buffers )
    return;

  // Perhaps a better idea would be each class having a 'name' attribute?
  if( sizeof(In)==sizeof(TimeSeries) && sizeof(Out)==sizeof(TimeSeries) ){
    TimeSeries* in = (TimeSeries*)input.ptr();
    TimeSeries* out = (TimeSeries*)output.ptr();
    
    in->swap_data( *out );
    if( free_scratch_space )
      out->resize(0);
  }
}

//! Prepends the output buffer with the saved data
template <class In, class Out>
uint64 dsp::Transformation<In, Out>::prepend_data(dsp::TimeSeries* ts_out){
  if( verbose )
    fprintf(stderr,"TRANS: (%s) entered prepend_data()\n",get_name().c_str());

  if( !ts_out )
    throw Error(InvalidState,"dsp::Transformation::prepend_data()",
		"BUG!  Valid data is saved, but the output is not a TimeSeries!");
  
  dsp::TimeSeries* sd = dynamic_cast<dsp::TimeSeries*>(saved_data.ptr());
  
  if( !sd )
    throw Error(InvalidState,"dsp::Transformation::prepend_data()",
		"Saved data buffer is not of type TimeSeries- this is not programmed for!  It is of type '%s'.  This classes name is '%s'",
		typeid(saved_data.ptr()).name(),get_name().c_str());

  if( debug ){
    unsigned twos = 0;
    for( unsigned ichan=0;ichan<ts_out->get_nchan();ichan++){
      for( unsigned ipol=0;ipol<ts_out->get_npol();ipol++){
	if( fabs(ts_out->get_datptr(ichan,ipol)[2]-2.0)<0.3 )
	  twos++;
      }
    }
    
    if( sd->get_nchan()>6 )
      fprintf(stderr,"TRANS::prepend_data() (%s) before operator= sd has samp [2]=%f [3]=%f twos=%d/%d sd->nchan=%d\n",
	      get_name().c_str(), sd->get_datptr(6,0)[2], sd->get_datptr(6,0)[3],
	      twos,sd->get_nchan()*sd->get_npol(),sd->get_nchan()); 
  }
  
  ts_out->operator=( *sd );

  if( debug ){
    unsigned twos = 0;
    for( unsigned ichan=0;ichan<ts_out->get_nchan();ichan++){
      for( unsigned ipol=0;ipol<ts_out->get_npol();ipol++){
	if( fabs(ts_out->get_datptr(ichan,ipol)[2]-2.0)<0.3 )
	  twos++;
      }
    }
    
    if( ts_out->get_nchan()>6 )
      fprintf(stderr,"TRANS::prepend_data() (%s) after operator= has samp [2]=%f [3]=%f twos=%d/%d nchan=%d\n",
	      get_name().c_str(), ts_out->get_datptr(6,0)[2], ts_out->get_datptr(6,0)[3],
	      twos,ts_out->get_nchan()*ts_out->get_npol(),ts_out->get_nchan()); 
  }

  uint64 samples_seeked = ts_out->get_ndat();
  ts_out->seek( ts_out->get_ndat() );
  
  ts_out->set_preserve_seeked_data( true );

  if( verbose )
    fprintf(stderr,"TRANS: (%s) prepend_data(): returning a prepend data buffer of ndat="UI64"\n",
	    get_name().c_str(),samples_seeked);

  return samples_seeked;
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

//! Save the last_nsamps into the saved_data buffer
template <class In, class Out>
void dsp::Transformation<In,Out>::save_data(uint64 last_nsamps){
  if( verbose )
    fprintf(stderr,"TRANS (%s) in save_data("UI64")\n",
	    get_name().c_str(),last_nsamps);

  if( !saved_data )
    saved_data = new Out;

  dsp::TimeSeries* ts = dynamic_cast<dsp::TimeSeries*>(saved_data.ptr());
  
  if( !ts )
    throw Error(InvalidState,"dsp::Transformation<In,Out>::save_data()",
		"Input is not a TimeSeries- this hasn't been coded for!");

  if( last_nsamps==0 ){
    ts->set_ndat(0);
    return;
  }

  dsp::TimeSeries* out = dynamic_cast<dsp::TimeSeries*>(get_output());

  ts->copy_configuration( out );
  ts->change_start_time( out->get_ndat() - last_nsamps );
  ts->resize(last_nsamps);

  if( debug ){
    for( unsigned ichan=0; ichan<out->get_nchan(); ichan++){
      for( unsigned ipol=0; ipol<out->get_npol(); ipol++){
	float* dat = out->get_datptr(ichan,ipol) + out->get_ndat() - last_nsamps;
	for( uint64 i=0; i<last_nsamps; i++)
	  dat[i] = float(i) + 0.1*float(ipol) + float(ichan)*0.0001;
      }
    }      
  }

  if( last_nsamps > out->get_ndat() )
    throw Error(InvalidState,"dsp::Transformation<In,Out>::save_data()",
		"last_nsamps provided ("UI64") is greater than output's ndat! (output's ndat="UI64")",
		last_nsamps, out->get_ndat());

  for( unsigned ichan=0; ichan<ts->get_nchan(); ichan++){
    for( unsigned ipol=0; ipol<ts->get_npol(); ipol++){
      float* to = ts->get_datptr(ichan,ipol);
      float* from = out->get_datptr(ichan,ipol) + out->get_ndat() - last_nsamps;

      for( uint64 i=0; i<last_nsamps; i++)
	to[i] = from[i];
    }
  }

  if( last_nsamps > 0 && debug ){
    unsigned twos = 0;
    for( unsigned ichan=0;ichan<ts->get_nchan();ichan++){
      for( unsigned ipol=0;ipol<ts->get_npol();ipol++){
	if( fabs(ts->get_datptr(ichan,ipol)[2]-2.0)<0.3 )
	  twos++;
      }
    }
    
    if( ts->get_nchan()>6)
      fprintf(stderr,"TRANS::save_data() (%s) at end got samples: [2]=%f [3]=%f twos=%d/%d\n",
	      get_name().c_str(),ts->get_datptr(6,0)[2], ts->get_datptr(6,0)[3],
	      twos,ts->get_nchan()*ts->get_npol());
  }

  valid_data_is_saved = true;
}

//! Returns how many samples were lost in the last call to operation()
template <class In, class Out>
int64 dsp::Transformation<In,Out>::get_input_samps_lost(){
  if( !has_input() )
    throw Error(InvalidState,"dsp::Transformation::get_input_samps_lost()",
		"No input defined");
  
  dsp::Observation* ts = (dsp::Observation*)dynamic_cast<const dsp::Observation*>( get_input() );
  if( !ts )
    throw Error(InvalidState,"dsp::Transformation::get_input_samps_lost()",
		"This method only works if the input is of type dsp::Observation");
  
  return input_samps_lost;
}

//! Decide whether you want to skip over samples that have been processed already (requires time_conserved==true; TimeSeries->TimeSeries)
template <class In, class Out>
void dsp::Transformation<In,Out>::set_process_samps_once(bool _process_samps_once){
  if( _process_samps_once == true && !get_time_conserved() )
    throw Error(InvalidState,"dsp::Transformation::set_process_samps_once()",
		"process_samps_once can't be set to true if time_conserved is false- it just hasn't been programmed for");
  
  process_samps_once = _process_samps_once; 
}

//! Not used now
template <class In, class Out>
void dsp::Transformation<In, Out>::prepend_output(dsp::TimeSeries* ts_out){
  fprintf(stderr,"Shouldn't enter here (dsp::Transformation<In, Out>::prepend_output())\n");
  abort();

  static int prepend_output_index = -1;

  if( prepend_output_index == -1 ){
    timers.push_back( OperationTimer("prepend_output") );  
    prepend_output_index = timers_index("prepend_output");
  }

  timers[prepend_output_index].start();

  if( !ts_out )
    throw Error(InvalidState,"dsp::Transformation::prepend_output()",
		"BUG!  Valid data is saved, but the output is not a TimeSeries!");

  if( !saved_data.ptr() )
    throw Error(InvalidState,"dsp::Transformation::prepend_output()",
		"Saved data buffer is null when valid_data_is_saved=%d\n",
		valid_data_is_saved);

  dsp::TimeSeries* sd = dynamic_cast<dsp::TimeSeries*>(saved_data.ptr());
  
  if( !sd )
    throw Error(InvalidState,"dsp::Transformation::prepend_output()",
		"Saved data buffer is not of type TimeSeries- this is not programmed for!  It is of type '%s'.  This classes name is '%s'",
		typeid(saved_data.ptr()).name(),get_name().c_str());

  if( sd->get_ndat()==0 ){
    timers[prepend_output_index].stop();
    return;
  }   

  if( sd->maximum_ndat() < sd->get_ndat() + ts_out->get_ndat() ){
    Reference::To<dsp::TimeSeries> temp(new dsp::TimeSeries);
    temp->copy_configuration( sd );
    temp->resize( sd->get_ndat() + ts_out->get_ndat() );
    temp->set_ndat( 0 );
    temp->append( sd );
    temp->swap_data( *sd );
  }
  sd->append( ts_out );
  //  fprintf(stderr,"TRANS 3. output ndat="UI64" sd ndat="UI64"\n",
  //  ts_out->get_ndat(),
  //  sd->get_ndat());
  
  sd->swap_data( *ts_out );
  sd->set_ndat( 0 );
  //  fprintf(stderr,"TRANS 4. output ndat="UI64" sd ndat="UI64"\n",
  //  ts_out->get_ndat(),
  //  sd->get_ndat());
  
  //  double secs_diff = (last_output_end - ts_out->get_start_time()).in_seconds();
  //int64 samps_diff = nint64(secs_diff * ts_out->get_rate());
  //if( samps_diff > 0 )
  //ts_out->seek( samps_diff );
  
  //  fprintf(stderr,"TRANS 5. output ndat="UI64" sd ndat="UI64"\n",
  //  ts_out->get_ndat(),
  //  sd->get_ndat());

  timers[prepend_output_index].stop();
}

#endif
