//-*-C++-*-

#ifndef __OutputBuffering_h
#define __OutputBuffering_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Buffers the Transformation output
  /*! DO NOT try to buffer the output of Transformations that accumulate
    their output - save_data will not work */
  class OutputBuffering : 
    public Transformation<TimeSeries,TimeSeries>::BufferingPolicy {
    
  public:
    
    //! Default constructor
    OutputBuffering (Transformation<TimeSeries,TimeSeries>* xform);

    //! Destructor
    ~OutputBuffering ();

    //! Set the output to be buffered
    void set_output (TimeSeries* output);

    //! Set the input
    void set_input (TimeSeries* input);
    
    //! Perform all buffering tasks required before transformation
    void pre_transformation ();
    
    //! Perform all buffering tasks required after transformation
    void post_transformation ();

    //! Returns the input start time (over-ridden by BandCombiner class)
    virtual MJD get_input_start_time ();

    //! Returns how many samples were lost
    virtual int64 get_input_samps_lost();

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

  protected:
    
    //! The next start sample
    uint64 next_start_sample;
    
    //! The output to be buffered
    Reference::To<TimeSeries> output;

    //! The input
    Reference::To<TimeSeries> input;
    
    //! The buffer
    Reference::To<TimeSeries> buffer;

    //! Start time of first operation
    MJD start_of_data;

    //! Set when the Transformation conserves time
    bool time_conserved;

    //! Valid data has been copied over into the buffer
    bool valid_data_is_saved;

    //! Save the last_nsamps into the buffer
    virtual void save_data (uint64 last_nsamps); 

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

    //! Works out what the MJD of the last sample processed was
    void workout_end_of_processed_data (MJD input_start_time,
					double time_prepended,
					double time_surplus);

    //! Rewinds 'end_of_processed_data' by the requested number of seconds
    void rewind_eopd (double seconds);

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
  
}

#endif // !defined(__OutputBuffering_h)
