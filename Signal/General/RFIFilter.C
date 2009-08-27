/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/RFIFilter.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Input.h"
#include "dsp/Bandpass.h"

#include "median_smooth.h"

using namespace std;

dsp::RFIFilter::RFIFilter ()
{
  calculated = false;
  nchan_bandpass = 1024;
  interval = 1.0;
  duty_cycle = 1.0;

  maximum_block_size = 8*1024;
  median_window = 51;
}

dsp::RFIFilter::~RFIFilter ()
{
}

//! Create an RFI filter with nchan channels
void dsp::RFIFilter::match (const Observation* obs, unsigned nchan)
{
  MJD epoch = obs->get_start_time();
  MJD input_start = input->get_input()->get_info()->get_start_time();

  if (verbose)
    cerr << "dsp::RFIFilter::match obs epoch=" << epoch
	 << " current start=" << start_time << " end=" << end_time << endl;

  if (calculated && epoch > start_time && epoch < end_time) {
    if (verbose)
      cerr << "dsp::RFIFilter::match use current filter" << endl;
    return;
  }

  double offset = (epoch - input_start).in_seconds();
  cerr << "dsp::RFIFilter::match input offset=" << offset << "s" << endl;

  if (offset < 0)
    throw Error (InvalidState, "dsp::RFIFilter::match",
		 "requested epoch=" + epoch.printdays(6) +
		 " precedes input data start=" + input_start.printdays(6));

  uint64_t ioffset = uint64_t (offset / interval);
  offset = ioffset * interval;

  start_time = input_start + offset;
  end_time = start_time + interval;

  if (!bandpass)
    bandpass = new Bandpass;

  if (!buffer)
    buffer = new WeightedTimeSeries;

  if (!data)
    data = new Response;

  if (verbose)
    cerr << "dsp::RFIFilter::match bandpass nchan=" << nchan_bandpass << endl;

  bandpass->set_input (buffer);
  bandpass->set_output (data);
  bandpass->set_nchan (nchan_bandpass);
  bandpass->reset_output();

  uint64_t position = input->get_input()->tell ();
  uint64_t blocksz = input->get_input()->get_block_size ();
  uint64_t overlap = input->get_input()->get_overlap ();
  TimeSeries* ptr = input->get_unpacker()->get_output();

  input->get_input()->seek (start_time);
  input->get_input()->set_block_size ( maximum_block_size );
  input->get_input()->set_overlap ( 0 );
  input->set_output (buffer);

  while (!input->get_input()->eod() &&
	 bandpass->get_integration_length() < interval) {
    input->operate ();
    bandpass->operate ();
  }

  input->get_input()->seek (position);
  input->get_input()->set_block_size (blocksz);
  input->get_input()->set_overlap (overlap);
  input->set_output (ptr);

  calculate (data);
}

void dsp::RFIFilter::calculate (Response* bp)
{
  unsigned nchan_bp = bp -> get_ndat();

  float* p0ptr = bp->get_datptr (0, 0);
  float* p1ptr = bp->get_datptr (0, 1);

  // form the total intensity bandpass
  vector<float> spectrum (nchan_bp);
  unsigned ichan=0;

  for (ichan=0; ichan < nchan_bp; ichan++)
    spectrum[ichan] = p0ptr[ichan]+p1ptr[ichan];

  fft::median_smooth (spectrum, median_window);

  double variance = 0.0;
  for (ichan=0; ichan < nchan_bp; ichan++)
  {
    spectrum[ichan] -= (p0ptr[ichan]+p1ptr[ichan]);
    spectrum[ichan] *= spectrum[ichan];
    // p0ptr[ichan] = spectrum[ichan];
    variance += spectrum[ichan];
  }

  variance /= nchan_bp;

  resize (1, 1, nchan_bp, 1);
  float* ptr = get_datptr(0,0);

  bool zapped = true;
  unsigned round = 1;
  unsigned total_zapped = 0;

  while (zapped)  {

    float cutoff = 16.0 * variance;
    cerr << "\tround " << round << " cutoff = " << cutoff << endl;

    zapped = false;
    round ++;

    for (ichan=0; ichan < nchan_bp; ichan++)
      if (spectrum[ichan] > cutoff ||
          (ichan && fabs(spectrum[ichan]-spectrum[ichan-1]) > 2*cutoff)) {
        variance -= spectrum[ichan]/nchan_bp;
        spectrum[ichan] = p0ptr[ichan] = p1ptr[ichan] = ptr[ichan] = 0.0;
	total_zapped ++;
        zapped = true; 
      }
      else
        ptr[ichan] = 1;

  }

  cerr << "\tzapped " << total_zapped << " channels" << endl;
  calculated = true;
}

//! Create an RFI filter with the same number of channels as Response
void dsp::RFIFilter::match (const Response* response)
{
  if (verbose)
    cerr << "dsp::RFIFilter::match Response nchan=" << response->get_nchan()
	 << " ndat=" << response->get_ndat() << endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() ) {

    if (verbose)
      cerr << "dsp::RFIFilter::match Response already matched" << endl;
    return;

  }

  unsigned required = response->get_nchan() * response->get_ndat();
  unsigned expand = required / get_ndat();
  unsigned shrink = get_ndat() / required;

  vector< complex<float> > phasors (required, 1.0);

  float* data = get_datptr(0,0);

  if (expand)
    for (unsigned idat=0; idat < get_ndat(); idat++)
      for (unsigned ip=0; ip < expand; ip++)
        phasors[idat*expand+ip] = data[idat];
  else if (shrink)
    for (unsigned idat=0; idat < required; idat++)
      for (unsigned ip=0; ip < expand; ip++)
        if (data[idat*shrink+ip] == 0.0)
          phasors[idat] = 0.0;
  else
    throw Error (InvalidState, "dsp::RFIFilter::match Response",
                 "not matched and not able to shrink or expand");

  set (phasors);

  resize (1, response->get_nchan(),
	  response->get_ndat(), 2);

}

//! Set the number of channels into which the band will be divided
void dsp::RFIFilter::set_nchan (unsigned nchan)
{

}

//! Set the interval over which the RFI mask will be calculated
void dsp::RFIFilter::set_update_interval (double seconds)
{

}

//! Set the fraction of the data used to calculate the RFI mask
void dsp::RFIFilter::set_duty_cycle (float cycle)
{

}

//! Set the source of the data
void dsp::RFIFilter::set_input (IOManager* _input)
{
  input = _input;
}

//! Set the buffer into which data will be read
void dsp::RFIFilter::set_buffer (TimeSeries* _buffer)
{
  buffer = _buffer;
}

//! Set the buffer into which the spectra will be integrated [optional]
void dsp::RFIFilter::set_data (Response* _data)
{
  data = _data;
}

//! Set the tool used to compute the spectra [optional]
void dsp::RFIFilter::set_bandpass (Bandpass* _bandpass)
{
  bandpass = _bandpass;
}
