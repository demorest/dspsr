#include "dsp/OutputBuffering.h"

dsp::OutputBufferBase::OutputBufferBase ()
{
  free_scratch_space = false;
  swap_buffers = false;
  input_samps_lost = 0;
  rounding = 0;
  valid_data_is_saved = false; 
  set_process_samps_once (false);
  end_of_processed_data = MJD::zero;
  samples_prepended = 0;
}

dsp::OutputBufferBase::~OutputBufferBase ()
{
}

//! Set the input
void dsp::OutputBufferBase::set_input (const Observation* _input)
{
  input = _input;
}

//! Set the output to be buffered
void dsp::OutputBufferBase::set_output (TimeSeries* _output)
{
  output = _output;
}

void dsp::OutputBufferBase::delete_saved_data()
{
  buffer = 0; 
}

//! Returns the input start time
MJD dsp::OutputBufferBase::get_input_start_time ()
{
  return input->get_start_time(); 
}

//! Perform all buffering tasks required before transformation
void dsp::OutputBufferBase::pre_transformation ()
{
  set_input();
  set_output();

  if( Operation::verbose )
    fprintf(stderr,"Entered dsp::OutputBufferBase::pre_transformation() (%s) this=%p input=%p\n",
	    get_parent_name().c_str(),this,input.get());

  time_in = input->get_duration();
  rate_in = input->get_rate();

  // Used by workout_end_of_processed_data() only
  // input_start_time is the earliest start time for dsp::BandCombiner
  input_start_time = get_input_start_time ();

  if (start_of_data == MJD::zero)
    start_of_data = input_start_time;

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::pre_transformation " << get_parent_name() << " input start time="
	 << (input_start_time-start_of_data).in_seconds() << " valid_data_is_saved=" << valid_data_is_saved << endl;

  if( valid_data_is_saved && Operation::verbose )
    fprintf(stderr,"dsp::OutputBufferBase::pre_transformation (%s) has a save_data buffer with ndat="UI64"\n",
	    get_parent_name().c_str(), buffer.ptr() ? buffer->get_ndat() : 0);

  if (valid_data_is_saved) {
    samples_prepended = prepend_data ();   
    if( Operation::verbose )
      fprintf(stderr,"dsp::OutputBufferBase::pre_transformation() (%s) just called prepend_data() to get samples_prepended="I64"\n",
	      get_parent_name().c_str(),samples_prepended);

    duration_prepended = samples_prepended / output->get_rate();
    time_in += duration_prepended;
  }
  else{
    duration_prepended = 0.0;
    samples_prepended = 0;
  }

  surplus_samples = seek_over_surplus_samps();

  if (surplus_samples < 0)
    throw Error (InvalidState, "dsp::OutputBufferBase::pre_transformation",
		 "Error calling seek_over_surplus_samps (%s)",
		 get_parent_name().c_str());
  
  if( Operation::verbose )
    fprintf(stderr,"dsp::OutputBufferBase::pre_transformation(%s) at end output is currently at a seekage of "UI64" samps\n",
	    get_parent_name().c_str(),output->get_seekage());  

  if( Operation::verbose )
    fprintf(stderr,"Exiting dsp::OutputBufferBase::pre_transformation() (%s) this=%p input=%p\n",
	    get_parent_name().c_str(),this,input.get());
}   

void dsp::OutputBufferBase::seek_back_over_surplus_samps ()
{
  const TimeSeries* ts_in = dynamic_cast<const TimeSeries*>(input.get());

  if (ts_in)
    const_cast<TimeSeries*>(ts_in)->seek (-surplus_samples);
}

//! Perform all buffering tasks required after transformation
void dsp::OutputBufferBase::post_transformation ()
{
  if (Operation::verbose)
  fprintf(stderr,"Entering dsp::OutputBufferBase::post_transformation() (%s) this=%p input=%p\n",
	  get_parent_name().c_str(),this,input.get());

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::post_transformation (" << get_parent_name() << ") START output ndat="
	 << output->get_ndat() << endl;

  if( Operation::verbose )
    fprintf(stderr,"dsp::OutputBufferBase::post_transformation(%s) at start output is currently at a seekage of "UI64" samps\n",
	    get_parent_name().c_str(),output->get_seekage());
  
  seek_back_over_surplus_samps ();

  rounding_stuff ();
  
  deprepend_data (double(surplus_samples)/rate_in);
  
  workout_end_of_processed_data (input_start_time,
				 double(samples_prepended)/output->get_rate(),
				 double(surplus_samples)/rate_in);
  
  swap_buffer_stuff();
  
  valid_data_is_saved = false;

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::post_transformation (" << get_parent_name() << ") END output ndat="
	 << output->get_ndat() << " start time="
	 << (output->get_start_time()-start_of_data).in_seconds() << endl;

  if (Operation::verbose)
    fprintf(stderr,"Exiting dsp::OutputBufferBase::post_transformation() (%s) this=%p input=%p\n",
	    get_parent_name().c_str(),this,input.get());
}

void dsp::OutputBufferBase::swap_buffer_stuff()
{
  if (!swap_buffers)
    return;

  const_cast<Observation*>(input.get())->swap_data (*output);

  if (free_scratch_space)
    output->resize(0);
}

uint64 dsp::OutputBufferBase::prepend_data ()
{
  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::prepend_data " << get_parent_name() << " START" << endl;
  
  output->operator= (*buffer);

  uint64 samples_seeked = output->get_ndat();
  output->seek (samples_seeked);
  output->set_preserve_seeked_data (true);

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::prepend_data copied " << samples_seeked
	 << " samples for " << get_parent_name() << endl;

  return samples_seeked;
}

//! Save the last_nsamps into the buffer buffer
void dsp::OutputBufferBase::save_data (uint64 last_nsamps)
{
  if (Operation::verbose)
    fprintf(stderr,"dsp::OutputBufferBase::save_data (%s) last_nsamps="UI64"\n",
	    get_parent_name().c_str(),last_nsamps);

  if (!buffer)
    buffer = new TimeSeries; 

  if( !last_nsamps ){
    buffer->set_ndat (0);
    return;
  }

  buffer->copy_configuration (output); // Gets weights
  buffer->Observation::operator= (*output); // Gets ndim
  buffer->change_start_time (output->get_ndat() - last_nsamps);
  buffer->resize(last_nsamps);

  if (last_nsamps > output->get_ndat())
    throw Error (InvalidState,"dsp::OutputBufferBase::save_data",
		"last_nsamps="UI64" > output ndat="UI64,
		last_nsamps, output->get_ndat());

  uint64 offset = (output->get_ndat() - last_nsamps) * output->get_ndim();
  uint64 count = last_nsamps * output->get_ndim() * sizeof(float);

  for (unsigned ichan=0; ichan<buffer->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<buffer->get_npol(); ipol++) {
      float* to = buffer->get_datptr(ichan,ipol);
      float* from = output->get_datptr(ichan,ipol) + offset;
      memcpy (to, from, count);
    }
  }

  valid_data_is_saved = true;
}

//! Rewinds 'end_of_processed_data' by the requested number of seconds
void dsp::OutputBufferBase::rewind_eopd (double seconds)
{
  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::rewind_eopd by " << seconds << "s for " << get_parent_name() << endl;
  end_of_processed_data -= seconds;
}

//! Handles the rounding stuff
void dsp::OutputBufferBase::rounding_stuff ()
{
  if (!rounding)
    return;

  uint64 old_ndat = output->get_ndat();
  output->set_ndat (output->get_ndat() - output->get_ndat()%rounding);

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::rounding_stuff has wiped "
	 << old_ndat - output->get_ndat() << " samples" << endl;
}

void dsp::OutputBufferBase::deprepend_data (double time_surplus)
{
  if( Operation::verbose )
    fprintf(stderr,"dsp::OutputBufferBase::deprepend_data() (%s) ready to seek back with samples_prepended="I64"\n",
	    get_parent_name().c_str(),samples_prepended);

  if (samples_prepended > 0) {
    output->seek (-samples_prepended);
    if (Operation::verbose)
      cerr << "dsp::OutputBufferBase::deprepend_data seeked back "
	   << samples_prepended << " samples; offset=" 
	   << output->get_samps_offset() << " for " << get_parent_name() << endl;
  }
  
  if (samples_prepended != 0)
    output->set_preserve_seeked_data (false);

  double time_out = output->get_duration();

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::deprepend_data " << get_parent_name() << 
      " time_in=" << time_in << " time_out=" << time_out <<
      " time_surplus=" << time_surplus << " so input_samps_lost=nint64("
	 << (time_in-time_surplus)-time_out << "*" << rate_in << ")" << endl;

  input_samps_lost = nint64 ((time_in-time_surplus-time_out)*rate_in);
}

//! Seeks over any samples that have already been processed
int64 dsp::OutputBufferBase::seek_over_surplus_samps ()
{
  const TimeSeries* ts_in = dynamic_cast<const TimeSeries*>(input.get());

  if (!ts_in || !process_samps_once || end_of_processed_data==MJD::zero 
      || time_conserved || input==output)
    return 0;
  
  MJD surplus_time = end_of_processed_data - input->get_start_time();
  double secs_surplus = surplus_time.in_seconds();
  int64 samps_surplus = nint64(secs_surplus * input->get_rate());
  
  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::seek_over_surplus_samps " << 
      samps_surplus << " samples for " << get_parent_name() << endl;

  if (samps_surplus < 0)
    throw Error(InvalidState,"dsp::OutputBufferBase::seek_over_surplus_samps",
		"Your last processing call ended at %f.\n\t"
		"This is %f seconds ("I64" samps) before current input starts",
		(end_of_processed_data-start_of_data).in_seconds(),
		fabs(secs_surplus), -samps_surplus);

  if (samps_surplus > int64(input->get_ndat())){
    cerr << "dsp::OutputBufferBase::seek_over_surplus_samps"
      " surpuls=" << samps_surplus << " > input ndat=" << input->get_ndat()
	 << " for " << get_parent_name() << endl;
    return -1;
  }

  const_cast<TimeSeries*>(ts_in)->seek (samps_surplus);

  return samps_surplus;
}

void
dsp::OutputBufferBase::workout_end_of_processed_data (MJD input_start_time,
						     double time_prepended,
						     double time_surplus)
{
  if (!time_conserved)
    return;
  
  end_of_processed_data = input_start_time + output->get_duration() 
    - time_prepended + time_surplus;

  if (Operation::verbose)
    cerr << "dsp::OutputBufferBase::workout_end_of_processed_data "
      "= input_start_time + out_dur - time_prepended + time_surplus "
      "= " << (input_start_time-start_of_data).in_seconds() << " + "
	 << output->get_duration() << " - " 
	 << time_prepended << " + "
	 << time_surplus <<
      " = " << (end_of_processed_data-start_of_data).in_seconds() <<
      " for " << get_parent_name() << endl;
}

int64 dsp::OutputBufferBase::get_input_samps_lost(){
  if( !input.ptr() )
    throw Error(InvalidState,"dsp::OutputBufferBase::get_input_samps_lost()",
		"No input defined (this=%s=%p='%s')",get_parent_name().c_str(),this,typeid(*this).name());
    
  return input_samps_lost;
}
