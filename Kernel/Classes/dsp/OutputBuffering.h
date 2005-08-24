//-*-C++-*-

#ifndef __OutputBuffering_h
#define __OutputBuffering_h

#include "environ.h"

namespace dsp {
  class OutputBufferBase;

  //! Enables OutputBuffering for a variety of Transformations
  template <class In>
  class OutputBuffering;
}

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Buffers the Transformation output
  /*! DO NOT try to buffer the output of Transformations that accumulate
    their output - save_data will not work */
  class OutputBufferBase : public BufferingPolicy {

  public:
    
    //! Default constructor
    OutputBufferBase ();

    //! Destructor
    virtual ~OutputBufferBase ();

    //! Set the output to be buffered
    void set_output (TimeSeries* output);

    //! Set the input
    void set_input (const Observation* input);
    
    //! Perform all buffering tasks required before transformation
    void pre_transformation ();
    
    //! Perform all buffering tasks required after transformation
    void post_transformation ();

    //! The start time of the next operation is essentially ignored
    void set_next_start (uint64) { }

    //! Set the minimum number of samples that can be processed
    void set_minimum_samples (uint64 minimum_samples) { }

    //! Returns the input start time (over-ridden by BandCombiner class)
    virtual MJD get_input_start_time ();

    //! Returns how many samples were lost
    virtual int64 get_input_samps_lost ();

    //! Save the last_nsamps into the buffer
    virtual void save_data (uint64 last_nsamps); 

    //! Rewinds 'end_of_processed_data' by the requested number of seconds
    void rewind_eopd (double seconds);

    //! Skip over samples that have already been processed
    /*! requires time_conserved==true */
    void set_process_samps_once (bool value = true)
    { process_samps_once = value; }

    bool get_process_samps_once () const 
    { return process_samps_once; }

    //! Set swap 'input' and 'output' before returning
    void set_swap_buffers (bool value = true) 
    { swap_buffers = value; }

    bool get_swap_buffers () const
    { return swap_buffers; }

    //! Delete the unused output buffer
    void set_free_scratch_space (bool value = true)
    { free_scratch_space = value; }

    bool get_free_scratch_space () const
    { return free_scratch_space; }

    //! Set the rounding factor
    void set_rounding (uint64 _rounding) { rounding = _rounding; }

    uint64 get_rounding() const { return rounding; }

    //! Returns the time prepended
    double get_duration_prepended() { return duration_prepended; }

    //! Returns parents name
    virtual string get_parent_name() = 0;

  protected:

    //! Derived classes set this from the parent
    virtual void set_output() = 0;
    virtual void set_input() = 0;
    
    //! The next start sample
    uint64 next_start_sample;
    
    //! The output to be buffered
    Reference::To<TimeSeries> output;

    //! The input
    Reference::To<const Observation> input;
    
    //! The buffer
    Reference::To<TimeSeries> buffer;

    //! Start time of first operation
    MJD start_of_data;

    //! Set when the Transformation conserves time
    bool time_conserved;

    //! Valid data has been copied over into the buffer
    bool valid_data_is_saved;

    //! Deletes the saved_data buffer
    virtual void delete_saved_data();

    //! The output ndat is rounded off to divide this number
    uint64 rounding;

    //! Handles the rounding stuff
    virtual void rounding_stuff ();

    //! The earliest start time for dsp::BandCombiner
    /*! Used by workout_end_of_processed_data() only */
    MJD input_start_time; 

    //! Duration of input data (in seconds)
    double time_in;

    //! Sampling rate of input data
    double rate_in;

    //! Stores how many samples were lost
    int64 input_samps_lost;

    //! Duration of prepended data (in seconds)
    double duration_prepended;

    //! Number of samples prepended during pre_transformation
    int64 samples_prepended;

    //! Number of samples in input that have already been processed
    int64 surplus_samples;

    //! Skip over samples that have already been processed
    bool process_samps_once;

    //! MJD of last point that was fully processed
    MJD end_of_processed_data;

    //! Seeks over any samples that have already been processed
    int64 seek_over_surplus_samps();

    //! Makes sure the input isn't changed by seeking over surplus samples
    void seek_back_over_surplus_samps ();

    //! Works out what the MJD of the last sample processed was
    void workout_end_of_processed_data (MJD input_start_time,
					double time_prepended,
					double time_surplus);

    //! Does all the swap buffer stuff
    void swap_buffer_stuff();

    //! Swap input and output buffers during post_transformation
    /*! You might set this to true when you have a class that must be
      outofplace, but you want your output to go into the same
      container as your input. */
    bool swap_buffers;

    //! If swap_buffers is true, then delete the output container buffers
    bool free_scratch_space;

    //! Prepends the output buffer with the saved data
    virtual uint64 prepend_data ();

    //! Allow the prepended data to be seen by the next Transformation
    virtual void deprepend_data (double time_surplus);

  };


  //! Enables OutputBuffering for a variety of Transformations
  template <class In>
  class OutputBuffering : public OutputBufferBase
  {

  public:

    //! Default constructor
    OutputBuffering (Transformation<In,TimeSeries>* xform);

    //! Perform all buffering tasks required before transformation
    void pre_transformation () { OutputBufferBase::pre_transformation(); }
    
    //! Perform all buffering tasks required after transformation
    void post_transformation () { OutputBufferBase::post_transformation(); }

    //! Returns parents name
    virtual string get_parent_name(){ return parent->get_name(); }

  protected:

    //! Set output from the parent
    virtual void set_output();

    //! Set input from the parent
    virtual void set_input();

    //! The parent
    Reference::To<Transformation<In,TimeSeries> > parent;

  };

  template<class In>
  void output_save (dsp::Transformation<In,dsp::TimeSeries>* tr, int64 ndat)
  {
    dsp::OutputBuffering<In>* policy = 0;

    if (tr->has_buffering_policy()) {
      BufferingPolicy* bp = tr->get_buffering_policy();
      policy = dynamic_cast<dsp::OutputBuffering<In>*>( bp );
    }

    if (!policy) {
      policy = new dsp::OutputBuffering<In>(tr);
      tr->set_buffering_policy( policy );
    }
    
    policy->save_data (ndat);
  }

}

template <class In>
void dsp::OutputBuffering<In>::set_input(){
  if( !parent->has_input() )
    throw Error(InvalidState,"dsp::OutputBuffering<In>::set_input()",
		"Parent has no input set!");
    
  input = parent->get_input();
}

template <class In>
void dsp::OutputBuffering<In>::set_output(){
  if( !parent->has_output() )
    throw Error(InvalidState,"dsp::OutputBuffering<In>::set_output()",
		"Parent has no output set!");

  output = parent->get_output();
}

template <class In>
dsp::OutputBuffering<In>::OutputBuffering (Transformation<In,TimeSeries>* tr)
{
  if (!tr)
    throw Error (InvalidParam, "dsp::OutputBuffering<In>",
		 "no Transformation");

  if (tr->has_input())
    input = tr->get_input();
  if (tr->has_output())
    output = tr->get_output();

  parent = tr;

  time_conserved = tr->get_time_conserved ();

  name = "OutputBuffering";
}

#endif // !defined(__OutputBuffering_h)

