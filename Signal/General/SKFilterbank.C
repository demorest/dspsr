/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKFilterbank.h"
#include "dsp/InputBuffering.h"

#include <errno.h>
#include <assert.h>
#include <string.h>

using namespace std;

dsp::SKFilterbank::SKFilterbank (unsigned _n_threads) : Filterbank ("SKFilterbank", anyplace)
{
  output_tscr = 0;

  tscrunch = 128;
  n_threads = _n_threads;

  if (n_threads < 1)
    throw Error (InvalidParam, "dsp::SKFilterbank::SKFilterbank",
     "nthreads < 1");

  context = new ThreadContext;
  state = Idle;

  thread_count = 0;
  ids.resize(n_threads);
  states.resize(n_threads);
  for (unsigned i=0; i<n_threads; i++)
  {
    states[i] = Idle;
    errno = pthread_create (&(ids[i]), 0, sk_thread, this);
    if (errno != 0)
      throw Error (FailedSys, "dsp::SKFilterbank", "pthread_create");
  }

  set_buffering_policy(new InputBuffering(this));
  debugd = 1;
}

dsp::SKFilterbank::~SKFilterbank () 
{
  if (verbose)
    cerr << "dsp::SKFilterbank::SKFilterbank~" << endl;
  stop_threads ();
  join_threads ();
}

void dsp::SKFilterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}

/*
 * These are preparations that could be performed once at the start of
 * the data processing
 */
void dsp::SKFilterbank::custom_prepare ()
{
  output->set_order( TimeSeries::OrderTFP );
  if (engine)
  {
    cerr << "dsp::SKFilterbank::custom_prepare engine->setup()" << endl;
    engine->setup();
    cerr << "dsp::SKFilterbank::custom_prepare engine->prepare (input, " << nsamp_fft << ")" << endl;
    engine->prepare (input, nsamp_fft);
  }
}

/* 
 * Return the number of samples required to increment the blocksize
 * such that the SKFB will have a full set of integrations
 */
uint64_t dsp::SKFilterbank::get_skfb_inc (uint64_t blocksize)
{
  if (verbose)
    cerr << "dsp::SKFilterbank::get_skfb_inc M=" << tscrunch 
         << " nsamp_fft=" << nsamp_fft << " blocksize=" << blocksize << endl;

  if (nsamp_fft == 0)
    throw Error (InvalidParam, "dsp::SKFilterbank::get_skfb_inc",
     "invalid nsamp_fft=0");

  uint64_t skfb_min = tscrunch * nsamp_fft;
  uint64_t remainder = blocksize % skfb_min;
  uint64_t increment = skfb_min - remainder;
  
  if (verbose)
    cerr << "dsp::SKFilterbank::get_skfb_inc skfb_min=" << skfb_min 
         << " remainder=" << remainder << " inc=" << increment << endl;

  return increment; 

}

void dsp::SKFilterbank::set_output_tscr (TimeSeries * _output_tscr)
{
  output_tscr = _output_tscr;
  output_tscr->set_order(TimeSeries::OrderTFP);
}

void dsp::SKFilterbank::filterbank ()
{

  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();

  if (npol != 2) 
  {
    throw Error (InvalidParam, "dsp::SKFilterbank::filterbank",
     "npol != 2");
    return;
  }

  if (verbose || debugd < 1)
    cerr << "dsp::SKFilterbank::filterbank input ndat=" << ndat
         << " npol=" << npol << " ndim=" << ndim 
         << " nsamp_fft=" << nsamp_fft << endl;

  const uint64_t nscrunches = ndat / (nsamp_fft * tscrunch);

  const uint64_t final_sample = nsamp_fft * nscrunches * tscrunch;

  if (has_buffering_policy())
  {
    if (debugd < 1)
      cerr << "dsp::SKFilterbank::filterbank setting next_start_sample=" 
           << final_sample << endl;
    get_buffering_policy()->set_next_start (final_sample);
  }

  // adjust tscr output
  if (output_tscr && nchan > output_tscr->get_nchan())
  {
    if (verbose)
      cerr << "dsp::SKFilterbank::filterbank output_tscr set_nchan(" << nchan << ")" << endl;
    output_tscr->set_ndat(1);
    output_tscr->set_nchan(nchan);
    output_tscr->set_npol(npol);
    output_tscr->set_ndim(1);
    output_tscr->resize(1);
  }

  if (engine)
  {
    engine->perform (input, output, output_tscr);
  }
  else
  {
    if (output_tscr && nchan > output_tscr->get_nchan())
    { 
      if (verbose)
        cerr << "dsp::SKFilterbank::filterbank S?_tscr.resize(" << nchan*npol << ")" << endl;
      S1_tscr.resize(nchan * npol);
      S2_tscr.resize(nchan * npol);
    }

    // initialise tscr
    if (output_tscr)
    {
      for (unsigned i=0; i<nchan*npol; i++)
      {
        S1_tscr[i]=0;
        S2_tscr[i]=0;
      }
    }
    if (verbose)
      cerr << "dsp::SKFilterbank::filterbank starting threads" << endl;

    // start SK threads
    start_threads();

    // wait for completion
    wait_threads();

    if (verbose)
      cerr << "dsp::SKFilterbank::filterbank threads ended" << endl;

    // now we need to combine the results from each SK Thread. Note
    // first thread should already be "in place"

    uint64_t decimated_ndat = ndat / (nsamp_fft * tscrunch);
    uint64_t thread_ndat = decimated_ndat / n_threads;
    uint64_t ndat_span = decimated_ndat / n_threads;
    uint64_t out_span = nchan * npol;
    uint64_t in_span  = nchan * npol * 2 * tscrunch;

    if (debugd < 1)
      cerr << "dsp::SKFilterbank::filterbank out_span=" << out_span << " in_span=" << in_span << endl;

    // the 0th thread will operate inplace, only need to memcpy the others
    for (unsigned ithread=1; ithread<n_threads; ithread++)
    {
      // last thread can have additional ndat
      if (ithread == n_threads - 1)
      {
        thread_ndat += decimated_ndat % n_threads;
      }

      uint64_t out_offset = ithread * ndat_span * out_span;
      uint64_t in_offset  = ithread * ndat_span * in_span;

      float *outdat = output->get_dattfp () + out_offset;
      float *indat  = output->get_dattfp () + in_offset;
      size_t size = thread_ndat * out_span * sizeof(float);

      if (debugd < 1)
        cerr << "dsp::SKFilterbank::filterbank [" << ithread << "] memcpy "
             << " out_offset=" << out_offset << " in_offset=" << in_offset
             << " ndat=" << thread_ndat * out_span << " size=" << size << endl;

      memcpy (outdat, indat, size);
    }

    // now compute the SK statisics for the tscr vector, from the S1 and S2 arrays
    if (debugd < 1)
      cerr << "dsp::SKFilterbank::filterbank calculating tscrunch SK estimates" << endl;

    if (output_tscr)
    {
      float S1 = 0;
      float S2 = 0;
      float M = (float) (tscrunch * decimated_ndat);
      float M_fac = (M+1) / (M-1);
      float * outdat = output_tscr->get_dattfp();

      if (debugd < 1)
        cerr << "dsp::SKFilterbank::filterbank tscr M=" << M <<" M_fac=" << M_fac << endl;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      { 
        // pol0 
        S1 = S1_tscr[2*ichan];
        S2 = S2_tscr[2*ichan];
        outdat[2*ichan] = M_fac * (M * (S2 / (S1*S1)) - 1);

        // pol1
        S1 = S1_tscr[2*ichan+1];
        S2 = S2_tscr[2*ichan+1];
        outdat[2*ichan+1] = M_fac * (M * (S2 / (S1*S1)) - 1);
      }
    }
  } 
  if (debugd < 1)
    cerr << "dsp::SKFilterbank::filterbank setting ndat=" << nscrunches << endl;
    
  output->set_ndat (nscrunches);
  output->set_npol (npol);
  output->set_state (Signal::PPQQ);
  output->set_ndim (1);

  uint64_t input_sample = input->get_input_sample();
  output->set_input_sample (input_sample / (nchan * tscrunch * (3 - ndim)));
  if (verbose || debugd < 1)
    cerr << "dsp::SKFilterbank::filterbank input_sample=" << input_sample 
         << ", output_sample=" << output->get_input_sample() << ", ndat=" 
         << output->get_ndat() << endl;

  double output_rate = input->get_rate() / (nchan * tscrunch * (3 - ndim));
  output->set_rate (output_rate);

  output->set_order( TimeSeries::OrderTFP );
  output->set_scale( 1.0 );

  if (debugd < 1)
    cerr << "dsp::SKFilterbank::filterbank done" << endl;
}

void* dsp::SKFilterbank::sk_thread (void* ptr)
{
  reinterpret_cast<SKFilterbank*>( ptr )->thread ();
  return 0;
}

void dsp::SKFilterbank::thread ()
{
  context->lock();

  // whichever thread gets here first will be thread_count (i.e. 0)
  unsigned thread_num = thread_count;
  thread_count++;

  // vertors for computing the thread tscr and fsrc S1/S2 statistics
  vector<float> thr_S1_tscr;
  vector<float> thr_S2_tscr;

  while (state != Quit)
  {
    // wait for Active state
    while (states[thread_num] == Idle)
    {
      context->wait ();
    }

    if (states[thread_num] == Quit)
    {
      context->unlock();
      return;
    }
    context->unlock();

    const uint64_t ndat = input->get_ndat();
    const unsigned npol = input->get_npol();
    const unsigned ndim = input->get_ndim();

    // each thread will process 1 / n_threads of the ndats, with the last
    // thread handling the remainder

    uint64_t thread_nscrunch = ndat / (nsamp_fft * tscrunch * n_threads);
    uint64_t thread_ndat =  thread_nscrunch * tscrunch * nsamp_fft;
    
    const uint64_t input_start_idat = thread_nscrunch * tscrunch * nsamp_fft * thread_num;
    const uint64_t input_offset     = input_start_idat * ndim;

#ifdef _DEBUG
    cerr << "thread[" << thread_num << "] INPUT start_idat=" 
         << input_start_idat << " offset=" << input_offset << endl;
#endif

    const uint64_t output_start_idat = thread_nscrunch * tscrunch * thread_num;
    const uint64_t output_offset     = output_start_idat * nchan * npol * 2;


#ifdef _DEBUG
    cerr << "thread[" << thread_num << "] OUTPUT start_idat=" 
         << output_start_idat << " offset=" << output_offset << endl;
#endif

    if (thread_num == n_threads - 1)
    {
      thread_ndat += ndat % (nsamp_fft * n_threads * tscrunch);
      thread_nscrunch = thread_ndat / (nsamp_fft * tscrunch);
    }

#ifdef _DEBUG
    cerr << "thread[" << thread_num << "] thread_ndat=" 
         << thread_ndat << " thread_nscrunch=" << thread_nscrunch << endl;
#endif

    // adjust the size of the tscr vectors
    if (thr_S1_tscr.size() < nchan * npol)
    {
      thr_S1_tscr.resize(nchan * npol);
      thr_S2_tscr.resize(nchan * npol);
    }

    // initialise vectors to 0
    for (unsigned i=0; i<nchan * npol; i++)
    {
      thr_S1_tscr[i]=0;
      thr_S2_tscr[i]=0;
    }

    const uint64_t thread_nffts = thread_ndat / nsamp_fft;

    uint64_t nfloat = nsamp_fft * ndim;

    // setup the correct initial offset for this thread
    float * outdat = output->get_dattfp () + output_offset;

#ifdef _DEBUG
    cerr << "thread[" << thread_num << "] FFT nffts=" << thread_nffts << endl;
    cerr << "thread[" << thread_num << "] SLD n=" << (nfloat * thread_nffts * npol) << " nfloat=" << nfloat << endl;
#endif

    for (uint64_t ifft=0; ifft<thread_nffts; ifft++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float* indat = input->get_datptr (0, ipol) + input_offset + ifft*nfloat;
        // FFT
        if (input->get_state() == Signal::Nyquist)
          forward->frc1d (nsamp_fft, outdat, indat);
        else
          forward->fcc1d (nsamp_fft, outdat, indat);

        // Square Law Detect
        for (unsigned i=0; i<nfloat; i+=2)
        {
          // Re squared
          outdat[i] *= outdat[i];
          // plus Im squared
          outdat[i] += outdat[i+1] * outdat[i+1];
          // pack the squared Pk (S2) into the complex hole
          outdat[i+1] = outdat[i] * outdat[i];
        }

        outdat += nfloat;
      }
    }

    // reset pointer to base address for this sthread
    outdat = output->get_dattfp() + output_offset;
    float * indat = outdat;     // inplace
    float * skoutdat = outdat;  // also inplace
    
    nfloat = nchan * 2 * npol;

    float S1;
    float S2;
    const float M = (float) tscrunch;
    const float M_fac = (M+1) / (M-1);

#ifdef _DEBUG
      cerr << "thread[" << thread_num << "] INT nscrunches=" << thread_nscrunch 
           << " nchan=" << nchan << " nfloat=" << nfloat << endl;
#endif

    for (uint64_t iscrunch=0; iscrunch<thread_nscrunch; iscrunch++)
    {
    
      // initialise accumulation results to 0
      for (uint64_t ifloat=0; ifloat < nfloat; ifloat+=2)
      {
        outdat[ifloat] = 0;
        outdat[ifloat+1] = 0;
      }
      
      // for the each fft, accumulate t_scrunch values
      for (uint64_t ifft=0; ifft < tscrunch; ifft++)
      {
        for (uint64_t ifloat=0; ifloat < nfloat; ifloat+=2)
        {
            // accumulate the S1 and S2 values
            outdat[ifloat] += indat[ifloat];
            outdat[ifloat+1] += indat[ifloat+1];
        }
        // 2 for complex and 2 for pol
        indat += nfloat;
      }

      // for each channel and pol calculate the SK estimator
      for (uint64_t ichan=0; ichan < nchan; ichan++)
      {
        // SK estimator for p0 packed into chan0, pol0
        S1 = outdat[2*ichan];
        S2 = outdat[2*ichan+1];
        skoutdat[2*ichan] = M_fac * (M * (S2 / (S1*S1)) - 1);

        thr_S1_tscr[2*ichan] += S1;
        thr_S2_tscr[2*ichan] += S2;

        // SK estimator for p1 packed into chan0, pol1
        S1 = outdat[(nchan*2) + (2*ichan)];
        S2 = outdat[(nchan*2) + (2*ichan) +1];
        skoutdat[2*ichan+1] = M_fac * (M * (S2 / (S1*S1)) - 1);

        thr_S1_tscr[2*ichan+1] += S1;
        thr_S2_tscr[2*ichan+1] += S2;
      }

      // have complex holes + pol
      outdat += nchan * 2 * npol;

      // no more complex holes 
      skoutdat += nchan * 2;
    }

    context->lock();
    if (states[thread_num] != Active)
      cerr << "thread[" << thread_num << "] state was not State != Active at end of processing loop" << endl;
    states[thread_num] = Idle;

    // update the tscr values
    for (unsigned ichan=0; ichan<nchan*npol; ichan++)
    {
      S1_tscr[ichan] += thr_S1_tscr[ichan];
      S2_tscr[ichan] += thr_S2_tscr[ichan];
    }

#ifdef _DEBUG
      cerr << "thread[" << thread_num << "] done" << endl;
#endif
    context->broadcast();
  }
  context->unlock();
}

void dsp::SKFilterbank::start_threads ()
{
  ThreadContext::Lock lock (context);

  while (state != Idle)
    context->wait ();

  for (unsigned i=0; i<n_threads; i++)
    states[i] = Active;
  state = Active;

  context->broadcast();
}

void dsp::SKFilterbank::wait_threads()
{
  ThreadContext::Lock lock (context);

  while (state == Active)
  {
    bool all_idle = true;
    for (unsigned i=0; i<n_threads; i++)
    {
      if (states[i] != Idle)
        all_idle = false;
    }

    if (all_idle)
    {
      state = Idle;
    }
    else
      context->wait ();
  }
}

void dsp::SKFilterbank::stop_threads ()
{
  ThreadContext::Lock lock (context);
  
  while (state != Idle)
    context->wait ();

  for (unsigned i=0; i<n_threads; i++)
    states[i] = Quit;
  state = Quit;

  context->broadcast();
}

void dsp::SKFilterbank::join_threads ()
{
  void * result = 0;  
  for (unsigned i=0; i<n_threads; i++)
    pthread_join (ids[i], &result);
}

