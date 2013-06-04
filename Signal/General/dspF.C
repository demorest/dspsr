/***************************************************************************
 *
 *   Copyright (C) 2012 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  dspF performs a forward FFT, writing spectral data to PSRDADA ring buffer
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/dsp.h"
#include "dsp/File.h"
#include "dsp/MultiFile.h"

#include "dsp/LoadToF1.h"
#include "dsp/LoadToFN.h"
#include "dsp/FilterbankConfig.h"

#include "CommandLine.h"
#include "FTransform.h"

#include <stdlib.h>

using namespace std;

void parse_options (int argc, char** argv);

void prepare (dsp::Pipeline* engine, dsp::Input* input);

// number of seconds to adjust clocks by
double offset_clock = 0.0;

// set the MJD
string mjd_string;

// set the telescope code
string telescope;

// bandwidth
double bandwidth = 0.0;

// centre_frequency
double centre_frequency = 0.0;

// The LoadToF configuration parameters
Reference::To<dsp::LoadToF::Config> config;

// names of data files to be processed
vector<string> filenames;


int main (int argc, char** argv) try
{
  config = new dsp::LoadToF::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;

  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToFN (config);
  else
    engine = new dsp::LoadToF (config);

  bool time_prep = dsp::Operation::record_time || config->get_cuda_ndevice();

  RealTimer preptime;
  if (time_prep)
    preptime.start();

  prepare (engine, config->open (argc, argv) );

  if (time_prep)
  {
    preptime.stop();
    cerr << "dspF: prepared in " << preptime << endl;
  }

  engine->run();
  engine->finish();
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void input_prepare (dsp::Input* input)
{
  dsp::Observation* info = input->get_info();

  if (bandwidth != 0)
  {
    cerr << "dspsr: over-riding bandwidth"
      " old=" << info->get_bandwidth() << " MHz"
      " new=" << bandwidth << " MHz" << endl;
    info->set_bandwidth (bandwidth);

    if (info->get_state() == Signal::Nyquist)
    {
      info->set_rate( fabs(bandwidth) * 2e6 );
      cerr << "dspsr: corrected Nyquist (real-valued) sampling rate="
           << info->get_rate() << " Hz" << endl;
    }
    else if (info->get_state () == Signal::Analytic)
    {
      info->set_rate( fabs(bandwidth) * 1e6 );
      cerr << "dspsr: corrected Analytic (complex-valued) sampling rate="
           << info->get_rate() << " Hz" << endl;
    }
  }

  if (centre_frequency != 0)
  {
    cerr << "dspsr: over-riding centre_frequency"
      " old=" << info->get_centre_frequency() <<
      " new=" << centre_frequency << endl;
    info->set_centre_frequency (centre_frequency);
  }

  if (!telescope.empty())
  {
    cerr << "dspsr: over-riding telescope code"
      " old=" << info->get_telescope() <<
      " new=" << telescope << endl;
    info->set_telescope (telescope);
  }

  if (!mjd_string.empty())
  {
    MJD mjd (mjd_string);
    cerr << "dspsr: over-riding start time"
      " old=" << info->get_start_time() <<
      " new=" << mjd << endl;
    info->set_start_time( mjd );
  }

  if (offset_clock)
  {
     MJD old = info->get_start_time();
     cerr << "dspsr: offset clock by " << offset_clock << " seconds" << endl;
     info->set_start_time( old + offset_clock );
  }
}

void prepare (dsp::Pipeline* engine, dsp::Input* input)
{
  config->input_prepare.set( input_prepare );

  engine->set_input( input );

  dsp::Observation* info = input->get_info();

  engine->prepare ();
}

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("dspF - convert dspsr input to fourier domain");
  menu.set_version ("dspF " + tostring(dsp::version) +
                    " <" + FTransform::get_library() + ">");

  config->add_options (menu);

  arg = menu.add (config->acc_len, 'a', "integrations");
  arg->set_help ("number of accumulations / integrations to use");

  arg = menu.add (config->nbits, 'b', "bits");
  arg->set_help ("number of bits to re-digitize spectral data");

  arg = menu.add (config->block_size, 'B', "MB");
  arg->set_help ("block size in megabytes");

  arg = menu.add (config->nbatch, 'n', "nbatch");
  arg->set_help ("Perform FFTs in batches");

  arg = menu.add (config->filterbank, 'F', "nchan");
  arg->set_help ("create a filterbank (voltages only)");
  arg->set_long_help
    ("Specify number of spectral channels; e.g. -F 8\n");

  arg = menu.add (config->hdu_key, 'k', "key");
  arg->set_help ("PSRDADA shared memory key");

  menu.parse (argc, argv);

  config->order = dsp::TimeSeries::OrderFPT;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& error)
{
  cerr << error.what() << endl;
  exit (-1);
}
