/***************************************************************************
 *
 *   Copyright (C) 2002-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Filterbank.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

#include "FTransform.h"

using namespace std;

// #define _DEBUG 1

dsp::Filterbank::Filterbank () : Convolution ("Filterbank", outofplace,true)
{
  nchan = 0;
  freq_res = 1;

  output_order = TimeSeries::OrderFPT;

  set_buffering_policy (new InputBuffering (this));
}

void dsp::Filterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}

void dsp::Filterbank::prepare ()
{
  make_preparations ();
  prepared = true;
}


/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::Filterbank::make_preparations ()
{
  if (nchan <= input->get_nchan() )
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		 "output nchan=%d <= input nchan=%d",
		 nchan, input->get_nchan());

  if (nchan % input->get_nchan() != 0)
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		 "output nchan=%d not a multiple of input nchan=%d",
		 nchan, input->get_nchan());

  //! Number of channels outputted per input channel
  nchan_subband = nchan / input->get_nchan();

  //! Complex samples dropped from beginning of cyclical convolution result
  nfilt_pos = 0;

  //! Complex samples dropped from end of cyclical convolution result
  nfilt_neg = 0;

  if (response)
  {
    if (verbose)
      cerr << "dsp::Filterbank call Response::match" << endl;

    // convolve the data with a frequency response function during
    // filterbank construction...

    response -> match (input, nchan);
    if (response->get_nchan() != nchan)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		   "response nchan=%d != output nchan=%d",
		   response->get_nchan(), nchan);

    nfilt_pos = response->get_impulse_pos ();
    nfilt_neg = response->get_impulse_neg ();

    freq_res = response->get_ndat();

    if (verbose)
      cerr << "dsp::Filterbank Response nfilt_pos=" << nfilt_pos 
	   << " nfilt_neg=" << nfilt_neg 
	   << " freq_res=" << response->get_ndat()
	   << " ndim=" << response->get_ndim() << endl;

    if (freq_res == 0)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		   "Response.ndat = 0");
  }

  // number of complex values in the result of the first fft
  unsigned n_fft = nchan_subband * freq_res;

  scalefac = 1.0;

  if (verbose)
  {
    string norm = "unknown";
    if (FTransform::get_norm() == FTransform::unnormalized)
      norm = "unnormalized";
    else if (FTransform::get_norm() == FTransform::normalized)
      norm = "normalized";
	
    cerr << "dsp::Filterbank::make_preparations n_fft=" << n_fft 
         << " freq_res=" << freq_res << " fft::norm=" << norm << endl;
  }

  if (FTransform::get_norm() == FTransform::unnormalized)
    scalefac = double(n_fft) * double(freq_res);

  else if (FTransform::get_norm() == FTransform::normalized)
    scalefac = double(n_fft) / double(freq_res);

  // number of complex samples invalid in result of small ffts
  nfilt_tot = nfilt_pos + nfilt_neg;

  // number of time samples by which big ffts overlap
  nsamp_overlap = 0;

  // number of time samples in first fft
  nsamp_fft = 0;

  if (input->get_state() == Signal::Nyquist)
  {
    nsamp_fft = 2 * n_fft;
    nsamp_overlap = 2 * nfilt_tot * nchan_subband;
  }
  else if (input->get_state() == Signal::Analytic)
  {
    nsamp_fft = n_fft;
    nsamp_overlap = nfilt_tot * nchan_subband;
  }

  else
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		 "invalid input data state = " + tostring(input->get_state()));

  // number of timesamples between start of each big fft
  nsamp_step = nsamp_fft - nsamp_overlap;

  if (verbose)
    cerr << "dsp::Filterbank::make_preparations nfilt_tot=" << nfilt_tot
         << " nsamp_fft=" << nsamp_fft << " nsamp_step=" << nsamp_step
         << " nsamp_overlap=" << nsamp_overlap << endl;

  // if given, test the validity of the window function
  if (apodization)
  {
    if( input->get_nchan() > 1 )
      throw Error(InvalidState,"dsp::Filterbank::make_preparations",
		  "not implemented for nchan=%d > 1",
		  input->get_nchan());

    if (apodization->get_ndat() != nsamp_fft)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		   "invalid apodization function ndat=%d"
		   " (nfft=%d)", apodization->get_ndat(), nsamp_fft);

    if (input->get_state() == Signal::Analytic 
	&& apodization->get_ndim() != 2)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		   "Signal::Analytic signal. Real apodization function.");

    if (input->get_state() == Signal::Nyquist 
	&& apodization->get_ndim() != 1)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		   "Signal::Nyquist signal. Complex apodization function.");
  }

  // matrix convolution
  matrix_convolution = false;

  if (response)
  {
    // if the response has 8 dimensions, then perform matrix convolution
    matrix_convolution = (response->get_ndim() == 8);

    if (verbose)
      cerr << "dsp::Filterbank::make_preparations with " 
	   << ((matrix_convolution)?"matrix":"complex") << " convolution"
	   << endl;

    if (matrix_convolution && input->get_nchan() > 1)
      throw Error(InvalidState,"dsp::Filterbank::make_preparations",
		  "matrix convolution untested for > one input channel");

    if (matrix_convolution && input->get_npol() != 2)
	throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		     "matrix convolution and input.npol != 2");
  }

  if (passband)
  {
    if (response)
      passband -> match (response);

    unsigned passband_npol = input->get_npol();
    if (matrix_convolution)
      passband_npol = 4;

    passband->resize (passband_npol, input->get_nchan(), n_fft, 1);

    if (!response)
      passband->match (input);
  }

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::Filterbank::make_preparations"
	" reserve=" << nsamp_fft << endl;

    get_buffering_policy()->set_minimum_samples (nsamp_fft);
  }

  // can support TFP only if freq_res == 1
  if (freq_res > 1)
    output_order = TimeSeries::OrderFPT;

  prepare_output ();

  if (engine)
  {
    engine->setup (nchan, freq_res, response->get_datptr(0,0));
    return;
  }

  using namespace FTransform;

  OptimalFFT* optimal = 0;
  if (response->has_optimal_fft())
    optimal = response->get_optimal_fft();

  if (optimal)
    FTransform::set_library( optimal->get_library( nsamp_fft ) );

  if (input->get_state() == Signal::Nyquist)
    forward = Agent::current->get_plan (nsamp_fft, FTransform::frc);
  else
    forward = Agent::current->get_plan (nsamp_fft, FTransform::fcc);

  if (optimal)
    FTransform::set_library( optimal->get_library( freq_res ) );

  if (freq_res > 1)
    backward = Agent::current->get_plan (freq_res, FTransform::bcc);

}

void dsp::Filterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::Filterbank::prepare_output set ndat=" << ndat << endl;

    output->set_npol( input->get_npol() );
    output->set_nchan( nchan );
    output->set_ndim( 2 );
    output->set_state( Signal::Analytic);
    output->resize( ndat );
  }

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());

  /* the problem: copy_configuration copies the weights array, which
     results in a call to resize_weights, which sets some offsets
     according to the reserve (for later prepend).  However, the
     offset is computed based on values that are about to be changed.
     This kludge allows the offsets to reflect the correct values
     that will be set later */

  unsigned tres_ratio = nsamp_fft / freq_res;

  if (weighted_output)
    weighted_output->set_reserve_kludge_factor (tres_ratio);

  output->copy_configuration ( get_input() );

  output->set_nchan( nchan );
  output->set_ndim( 2 );
  output->set_state( Signal::Analytic );
  output->set_order( output_order );

  if (weighted_output)
  {
    weighted_output->set_reserve_kludge_factor (1);
    weighted_output->convolve_weights (nsamp_fft, nsamp_step);
    weighted_output->scrunch_weights (tres_ratio);
  }

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::Filterbank::prepare_output reset ndat=" << ndat << endl;
    output->resize (ndat);
  }
  else
  {
    ndat = input->get_ndat() / tres_ratio;

    if (verbose)
      cerr << "dsp::Filterbank::prepare_output scrunch ndat=" << ndat << endl;
    output->resize (ndat);
  }

  if (verbose)
    cerr << "dsp::Filterbank::prepare_output output ndat="
         << output->get_ndat() << endl;

  output->rescale (scalefac);
  
  if (verbose) cerr << "dsp::Filterbank::prepare_output scale="
                    << output->get_scale() <<endl;

  /*
   * output data will have new sampling rate
   * NOTE: that nsamp_fft already contains the extra factor of two required
   * when the input TimeSeries is Signal::Nyquist (real) sampled
   */
  double ratechange = double(freq_res) / double (nsamp_fft);
  output->set_rate (input->get_rate() * ratechange);

  if (freq_res == 1)
    output->set_dual_sideband (true);

  /*
   * if freq_res is even, then each sub-band will be centred on a frequency
   * that lies on a spectral bin *edge* - not the centre of the spectral bin
   */
  output->set_dc_centred (freq_res%2);

#if 0
  // the centre frequency of each sub-band will be offset
  double channel_bandwidth = input->get_bandwidth() / nchan;
  double shift = double(freq_res-1)/double(freq_res);
  output->set_centre_frequency_offset ( 0.5*channel_bandwidth*shift );
#endif

  // dual sideband data produces a band swapped result
  if (input->get_dual_sideband())
    output->set_swap (true);

  // increment the start time by the number of samples dropped from the fft
  output->change_start_time (nfilt_pos);

  if (verbose)
    cerr << "dsp::Filterbank::prepare_output start time += "
	 << nfilt_pos << " samps -> " << output->get_start_time() << endl;

  // enable the Response to record its effect on the output Timeseries
  if (response)
    response->mark (output);
}

void dsp::Filterbank::reserve ()
{
  resize_output (true);
}

void dsp::Filterbank::resize_output (bool reserve_extra)
{
  const uint64_t ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::Filterbank::reserve input ndat=" << ndat << endl;

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  this->npart = 0;

  if (ndat > nsamp_overlap)
    npart = (ndat-nsamp_overlap)/nsamp_step;

  // on some iterations, ndat could be large enough to fit an extra part
  if (reserve_extra)
    npart ++;

  // points kept from each small fft
  unsigned nkeep = freq_res - nfilt_tot;

  uint64_t output_ndat = npart * nkeep;

  // prepare the output TimeSeries
  prepare_output (output_ndat, true);
}

void set_pointers (dsp::Filterbank::Engine* engine, dsp::TimeSeries* output, 
                   unsigned ipol, uint64_t out_offset)
{
  const unsigned nchan = output->get_nchan();
  engine->output_ptr.resize (nchan);

  for (unsigned ichan=0; ichan < nchan; ichan++)
    engine->output_ptr[ichan] = output->get_datptr (ichan, ipol) + out_offset;
}

void dsp::Filterbank::transformation ()
{
  if (verbose)
    cerr << "dsp::Filterbank::transformation input ndat=" << input->get_ndat()
	 << " nchan=" << input->get_nchan() << endl;

  if (!prepared)
    prepare ();

  resize_output ();

  if (has_buffering_policy())
    get_buffering_policy()->set_next_start (nsamp_step * npart);

  uint64_t output_ndat = output->get_ndat();

  // points kept from each small fft
  unsigned nkeep = freq_res - nfilt_tot;

  if (verbose)
    cerr << "dsp::Filterbank::transformation npart=" << npart 
	 << " nkeep=" << nkeep << " output_ndat=" << output_ndat << endl;

  // set the input sample
  int64_t input_sample = input->get_input_sample();
  if (output_ndat == 0)
    output->set_input_sample (0);
  else if (input_sample >= 0)
    output->set_input_sample ((input_sample / nsamp_step) * nkeep);

  if (verbose)
    cerr << "dsp::Filterbank::transformation after prepare output"
            " ndat=" << output->get_ndat() << 
            " input_sample=" << output->get_input_sample() << endl;

  if (!npart)
  {
    if (verbose)
      cerr << "dsp::Filterbank::transformation empty result" << endl;
    return;
  }

  if (freq_res == 1 && output_order == TimeSeries::OrderTFP)
  {
    if (verbose)
      cerr << "dsp::Filterbank::transformation TFP filterbank" << endl;
    tfp_filterbank ();
    return;
  }

  // initialize scratch space for FFTs
  unsigned bigfftsize = nchan_subband * freq_res * 2;
  if (input->get_state() == Signal::Nyquist)
    bigfftsize += 8;

  // also need space to hold backward FFTs
  unsigned scratch_needed = bigfftsize + 2 * freq_res;

  if (apodization)
    scratch_needed += bigfftsize;

  if (matrix_convolution)
    scratch_needed += bigfftsize;

  // divide up the scratch space
  float* c_spectrum[2];
  c_spectrum[0] = scratch->space<float> (scratch_needed);
  c_spectrum[1] = c_spectrum[0];
  if (matrix_convolution)
    c_spectrum[1] += bigfftsize;

  float* c_time = c_spectrum[1] + bigfftsize;
  float* windowed_time_domain = c_time + 2 * freq_res;

  unsigned cross_pol = 1;
  if (matrix_convolution)
    cross_pol = 2;

  if (verbose)
    cerr << "dsp::Filterbank::transformation enter main loop" <<
      " cpol=" << cross_pol << " npol=" << input->get_npol() << endl;

  // number of floats to step between input to filterbank
  const unsigned long in_step = nsamp_step * input->get_ndim();

  // number of floats to step between output from filterbank
  const unsigned long out_step = nkeep * 2;

  // counters
  unsigned ipt, ipol, jpol, ichan;
  uint64_t ipart;

  const unsigned npol = input->get_npol();

  // offsets into input and output
  uint64_t in_offset, out_offset;

  // some temporary pointers
  float* time_dom_ptr = NULL;  
  float* freq_dom_ptr = NULL;

  // do a 64-bit copy
  uint64_t* data_into = NULL;
  uint64_t* data_from = NULL;

  if (engine)
  {
    engine->scratch = c_spectrum[0];
    engine->nfilt_pos = nfilt_pos;
    engine->freq_res = freq_res;
    engine->nkeep = nkeep;
  }

  for (unsigned input_ichan=0; input_ichan<input->get_nchan(); input_ichan++)
  {
    if (engine)
    {
      for (ipol=0; ipol < npol; ipol++)
      {
	for (ipart=0; ipart<npart; ipart++)
	{
#ifdef _DEBUG
	  cerr << "ipart=" << ipart << endl;
#endif
	  in_offset = ipart * in_step;
	  out_offset = ipart * out_step;
      
	  time_dom_ptr = const_cast<float*>(input->get_datptr (input_ichan, ipol)) + in_offset;

	  set_pointers (engine, output, ipol, out_offset);
	  
	  engine->perform (time_dom_ptr);

	} // for each part

      } // for each polarization

    }
    else // not using engine
    {
      for (ipart=0; ipart<npart; ipart++)
      {
#ifdef _DEBUG
	cerr << "ipart=" << ipart << endl;
#endif

	in_offset = ipart * in_step;
	out_offset = ipart * out_step;
      
	for (ipol=0; ipol < npol; ipol++)
	{
	  for (jpol=0; jpol<cross_pol; jpol++)
	  {
	    if (matrix_convolution)
	      ipol = jpol;
	    
	    time_dom_ptr = const_cast<float*>(input->get_datptr (input_ichan, ipol));

	    time_dom_ptr += in_offset;
	    
	    if (apodization)
	    {
	      apodization -> operate (time_dom_ptr, windowed_time_domain);
	      time_dom_ptr = windowed_time_domain;
	    }
	    if (input->get_state() == Signal::Nyquist)
	      forward->frc1d (nsamp_fft, c_spectrum[ipol], time_dom_ptr);
	    else
	      forward->fcc1d (nsamp_fft, c_spectrum[ipol], time_dom_ptr);
	    
	    
	  }
	  
	  if (matrix_convolution)
	  {

	    if (passband)
	      passband->integrate (c_spectrum[0], c_spectrum[1], input_ichan);

	    // cross filt can be set only if there is a response
	    response->operate (c_spectrum[0], c_spectrum[1]);

	  }
	  else
	  {
	    if (passband)
	      passband->integrate (c_spectrum[ipol], ipol, input_ichan);

	    if (response)
	      response->operate (c_spectrum[ipol], ipol,
				 input_ichan*nchan_subband, nchan_subband);
	  }

	  for (jpol=0; jpol<cross_pol; jpol++)
          {
	    if (matrix_convolution)
	      ipol = jpol;
	    
	    if (freq_res == 1)
            {
	      data_from = (uint64_t*)( c_spectrum[ipol] );
	      for (ichan=0; ichan < nchan_subband; ichan++)
              {
		data_into = (uint64_t*)( output->get_datptr (input_ichan*nchan_subband+ichan, ipol) + out_offset );
		
		*data_into = data_from[ichan];
	      }
	      continue;
	    }


	    // freq_res > 1 requires a backward fft into the time domain
	    // for each channel

            unsigned jchan = input_ichan * nchan_subband;
            freq_dom_ptr = c_spectrum[ipol];

	    for (ichan=0; ichan < nchan_subband; ichan++)
            {

	      backward->bcc1d (freq_res, c_time, freq_dom_ptr);

	      freq_dom_ptr += freq_res*2;
	      
	      data_into = (uint64_t*)( output->get_datptr (jchan+ichan, ipol) + out_offset);
	      data_from = (uint64_t*)( c_time + nfilt_pos*2 );  // complex nos.

	      for (ipt=0; ipt < nkeep; ipt++)
	        data_into[ipt] = data_from[ipt];
	      
	    } // for each output channel
	    
	  } // for each cross poln
	
	} // for each polarization
      
      } // for each big fft (ipart)
    
    } // if not using engine

  } // for each input channel


  if (verbose)
    cerr << "dsp::Filterbank::transformation return with output ndat="
	 << output->get_ndat() << endl;
}

void dsp::Filterbank::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

void dsp::Filterbank::tfp_filterbank ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned input_ichan = 0;

  if (verbose)
    cerr << "dsp::Filterbank::tfp_filterbank input ndat=" << ndat << endl;

  // number of FFTs
  const uint64_t npart = ndat / nsamp_fft;
  const unsigned long nfloat = nsamp_fft * input->get_ndim();

  float* outdat = output->get_dattfp ();

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    const float* indat = input->get_datptr (input_ichan, ipol);

    for (uint64_t ipart=0; ipart < npart; ipart++)
    {
      if (input->get_state() == Signal::Nyquist)
	forward->frc1d (nsamp_fft, outdat, indat);
      else
        forward->fcc1d (nsamp_fft, outdat, indat);

      outdat += nfloat;
      indat += nfloat;
    }
  }

  if (npol == 2)
  {
    /* the data are now in TPF order, whereas TFP is desired.
       so square law detect, then pack p1 into the p0 holes */

    uint64_t nfloat = npart * npol * nchan;
    outdat = output->get_dattfp ();

    if (verbose)
      cerr << "dsp::Filterbank::tfp_filterbank detecting" << endl;

    for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
    {
      // Re squared
      outdat[ifloat*2] *= outdat[ifloat*2];
      // plus Im squared
      outdat[ifloat*2] += outdat[ifloat*2+1] * outdat[ifloat*2+1];
    }

    if (verbose)
      cerr << "dsp::Filterbank::tfp_filterbank interleaving" << endl;

    nfloat = npart * nchan;

    for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
    {
      // set Im[p0] = Re[p1]
      outdat[ifloat*2+1] = outdat[ifloat*2+nfloat];
    }

    output->set_state (Signal::PPQQ);
    output->set_ndim (1);
  }
}

#if 0

void filterbank::scattered_power_correct (float_Stream& dispersed_power,
					  const a2d_correct& digitization)
{
  if (!ppweight)
    throw string ("filterbank::scattered_power_correct "
	       "ERROR: no time weights");
  
  // check the validity of this transformation
  if (dispersed_power.get_state() != Detected)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power must be detected");

  if (dispersed_power.rate != rate)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power must have same sampling rate");

  if (dispersed_power.start_time > start_time)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power does not start early enough");

  if (dispersed_power.end_time < end_time)
    throw string ("filterbank::scattered_power_correct "
	       "dispersed power ends too soon");

  if (verbose)
    cerr << "filterbank::scattered_power_correct " << endl
	 << " start:" << start_time << " end:" << end_time << endl
	 << " dp.start:" << dispersed_power.start_time
	 << " dp.end:" << dispersed_power.end_time  << endl;

  double offset_time = (start_time - dispersed_power.start_time).in_seconds();
  unsigned offset_samples = (unsigned) floor (offset_time * rate + 0.5);

  if (verbose)
    cerr << "filterbank::scattered_power_correct "
	 << "offset (us):" << offset_time * 1e6 
	 << " offset (samples):" << offset_samples << endl;

  // sanity check
  if (dispersed_power.ndat - offset_samples < ndat)
    throw Error (InvalidState, "filterbank::scattered_power_correct "
	       " dp.ndat="I64" < ndat="I64" + offset="I64,
	       dispersed_power.ndat, ndat, offset_samples);
  
  // only PP and QQ are corrected...
  int cpol = 2;
  if (npol == 1)
    // ...unless only Signal::Stokes I remains
    cpol = 1;

  if (dispersed_power.npol != cpol)
    throw Error (InvalidState, "filterbank::scattered_power_correct "
	       "dispersed power must have npol=%d", cpol);

  if (!( (get_state() == Detected) || (get_state() == Signal::Coherence) ))
    throw string ("filterbank::scattered_power_correct invalid state="
		  + state_str());

  double normalize = scale / dispersed_power.scale;

  if (verbose)
    cerr << "filterbank::scattered_power_correct "
	 << "scale:" << scale << " dp.scale:" << dispersed_power.scale
	 << " normalize:" << normalize << endl
	 << " correct " << cpol
	 << " polns by " << nchan << " chans by " << ndat << " "
	 << state_str() << " pts" << endl;

  int vincr = 0;    // steps between subsequent time samples in dispersed power
  float* vptr = 0;  // points to ipol-T0 in the dispersed power

  double cfac = 0.0;

  int ipol;
  Int64 ipt, endpt;
    
  for (ipol=0; ipol < cpol; ipol++) {

    vptr = dispersed_power.datptr (0, ipol, vincr);
    vptr += offset_samples * vincr;

    ipt = 0;
    endpt = ppweight;

    for (unsigned iwt=0; iwt<nweights; iwt++) {

      if (weights[ipol][iwt] == 0)
	cfac = 0;
      else
	// the nchan denominator is absorbed in the dispersed_power scale
	cfac = (1.0 - digitization.spc_factor(weights[ipol][iwt])) * normalize;
      
      if (endpt > ndat)
	endpt = ndat;
      
      // set the float_Stream to equal the scattered power correction
      for (; ipt<endpt; ipt++) {
	*vptr *= cfac;
	vptr += vincr;
      }

      endpt += ppweight;

    } // for each weight
  
    // sanity check
    if (ipt != ndat)
      throw Error (InvalidState, "filterbank::scattered_power_correct\n"
		 " sanity check ipt="I64" should equal ndat="I64, ipt, ndat);

  } // for each polarization
  
  register float* vp = 0;
  register float* fp = 0; // points to F0-ipol-T0 in the filterbank
  register int fincr = 0;   // step between subsequent time samples in filterbank

  for (ipol=0; ipol < cpol; ipol++) {

    vptr = dispersed_power.datptr (0, ipol, vincr);
    vptr += offset_samples * vincr;

    for (int ichan=0; ichan < nchan; ichan++) {

      fp = datptr (ichan, ipol, fincr);
      vp = vptr;

      for (ipt=0; ipt<ndat; ipt++)  {
	*fp -= *vp;
	fp += fincr;
	vp += vincr;
      }
 
    } // for each channel
    
  } // for each polarization

}

void filterbank::Hanning (int degree)
{
  if (state < Detected)
    throw string ("filterbank::Hanning called on undetected data");

  SignalProcessing::Window triangle;

  // construct the parzen window triangle for real data
  triangle.Parzen ((degree-1) * 2 + 1, false);
  triangle.normalize();

  scrunch (triangle);
}

#endif
