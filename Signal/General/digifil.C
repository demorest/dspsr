/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digifil converts any file format recognized by dspsr into sigproc
  filterbank (.fil) format.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFil.h"

#include "CommandLine.h"
#include "FTransform.h"

#include <stdlib.h>

using namespace std;

// The LoadToFil configuration parameters
Reference::To<dsp::LoadToFil::Config> config;

// names of data files to be processed
vector<string> filenames;

void parse_options (int argc, char** argv);

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFil::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine = new dsp::LoadToFil (config);

  engine->set_input( config->open (argc, argv) );
  engine->prepare ();   
  engine->run();
  engine->finish();
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("digifil - convert dspsr input to search mode output");
  menu.set_version ("digifil " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  config->add_options (menu);

  arg = menu.add (config->nbits, 'b', "bits");
  arg->set_help ("number of bits per sample output to file");

  arg = menu.add (config->block_size, 'B', "MB");
  arg->set_help ("block size in megabytes");

  arg = menu.add (config->rescale_constant, 'c');
  arg->set_help ("keep offset and scale constant");

  arg = menu.add (config->filterbank_nchan, 'F', "nchan");
  arg->set_help ("create a filterbank (voltages only)");

  arg = menu.add (config->frequency_resolution, 'x', "nfft");
  arg->set_help ("backward FFT length in voltage filterbank");

  arg = menu.add (config->dedisperse, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  arg = menu.add (config->dispersion_measure, 'D', "dm");
  arg->set_help (" set the dispersion measure");

  arg = menu.add (config->tscrunch_factor, 't', "nsamp");
  arg->set_help ("decimate in time");

  arg = menu.add (config->fscrunch_factor, 'f', "nchan");
  arg->set_help ("decimate in frequency");

  arg = menu.add (config->rescale_seconds, 'I', "secs");
  arg->set_help ("rescale interval in seconds");

  arg = menu.add (config->output_filename, 'o', "file");
  arg->set_help ("output filename");

  bool revert = false;
  arg = menu.add (revert, 'p');
  arg->set_help ("revert to FPT order");

  menu.parse (argc, argv);

  if (revert)
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

