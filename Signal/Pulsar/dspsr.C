/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/dsp.h"
#include "dsp/File.h"
#include "dsp/MultiFile.h"

#include "dsp/LoadToFoldConfig.h"
#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldN.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Parameters.h"
#include "Pulsar/Predictor.h"

#include "FTransform.h"

#include "load_factory.h"
#include "dirutil.h"

#include <iostream>

#include <stdlib.h>
#include <errno.h>
#include <string.h>

using namespace std;

void parse_options (int argc, char** argv);

static bool verbose = false;

// sets up the pipeline using the following attributes

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
// Pulsar name
string pulsar_name;

// Specified command line options that apply to baseband data only
string baseband_options;

// The LoadToFold configuration parameters
Reference::To<dsp::LoadToFold::Config> config;

// Number of threads used to process the data
unsigned nthread = 0;



// names of data files to be processed
vector<string> filenames;

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFold::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;

  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToFoldN (config);
  else
    engine = new dsp::LoadToFold (config);

  bool time_prep = dsp::Operation::record_time || config->get_cuda_ndevice();

  unsigned nfile = filenames.size();
  if (config->continuous_obs) 
    nfile = 1;

  for (unsigned ifile=0; ifile < nfile; ifile++) try
  {
    if (verbose)
    {
      if (config->continuous_obs)
      {
        cerr << "opening data files: " << endl;
        for (unsigned ii=0; ii < filenames.size(); ii++)
          cerr << "  " << filenames[ii] << endl;
      }
      else
        cerr << "opening data file " << filenames[ifile] << endl;
    }
    
    RealTimer preptime;
    if (time_prep)
      preptime.start();

    dsp::File *file;
    if (config->continuous_obs) 
    {
      dsp::MultiFile *multi = new dsp::MultiFile;
      multi->open (filenames);
      file = multi;
    }
    else
      file = dsp::File::create( filenames[ifile] );

    prepare (engine, file);

    if (time_prep)
    {
      preptime.stop();
      cerr << "dspsr: prepared in " << preptime << endl;
    }

    if (verbose)
    {
      if (config->continuous_obs)
        cerr << "data files opened" << endl;
      else
        cerr << "data file " << filenames[ifile] << " opened" << endl;
    }

    engine->run();
    engine->finish();
  }
  catch (Error& error)
  {
    cerr << error << endl;
    return -1;
  }

  return 0;
}

catch (Error& error)
{
  cerr << "Error thrown: " << error << endl;
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
  
  if (!pulsar_name.empty())
  {
    cerr << "dspsr: over-riding source name"
      " old=" << info->get_source() <<
      " new=" << pulsar_name << endl;
    info->set_source( pulsar_name );   
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

  info->set_dispersion_measure (config->dispersion_measure);
}

void prepare (dsp::Pipeline* engine, dsp::Input* input)
{
  config->input_prepare.set( input_prepare );

  engine->set_input( input );

  dsp::Observation* info = input->get_info();

  if (info->get_detected() && !baseband_options.empty())
    throw Error (InvalidState, "prepare",
		 "input type " + input->get_name() +
		 " yields detected data and the command line option(s):"
		 "\n\n" + baseband_options + "\n\n"
		 " are specific to baseband (undetected) data.");

  engine->prepare ();    
}

#include "CommandLine.h"

void parse_options (int argc, char** argv) try
{

  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("dspsr - digital signal processing of pulsar signals");
  menu.set_version ("dspsr " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  /* ***********************************************************************

  General Processing Options
  
  *********************************************************************** */

  menu.add ("\n" "General processing options:");

  string ram_min;
  arg = menu.add (ram_min, "minram", "MB");
  arg->set_help ("minimum RAM usage in MB");

  string ram_limit;
  arg = menu.add (ram_limit, 'U', "MB|minX");
  arg->set_help ("upper limit on RAM usage");
  arg->set_long_help
    ("specify either the floating point number of megabytes; e.g. -U 256 \n"
     "or a multiple of the minimum possible block size; e.g. -U minX2 \n");

  config->add_options (menu);

  /* ***********************************************************************

  File Handling Options
  
  *********************************************************************** */

  menu.add ("\n" "File handling options:");

  vector<string> unpack;
  arg = menu.add (unpack, '2', "code");
  arg->set_help ("unpacker options (\"2-bit\" excision)");
  arg->set_long_help
    (" -2c<cutoff>    threshold for impulsive interference excision \n"
     " -2n<sample>    number of samples used to estimate undigitized power \n"
     " -2t<threshold> two-bit sampling threshold at record time \n");


  arg = menu.add (config->continuous_obs, "cont");
  arg->set_help ("treat input files as a single continuous observation");


  /* ***********************************************************************

  Source Options
  
  *********************************************************************** */

  menu.add ("\n" "Source options:");

  arg = menu.add (bandwidth, 'B', "bandwidth");
  arg->set_help ("set the bandwidth in MHz");

  arg = menu.add (centre_frequency, 'f', "frequency");
  arg->set_help ("set the centre frequency in MHz");

  arg = menu.add (telescope, 'k', "telescope");
  arg->set_help ("set the telescope name");
    
  arg = menu.add (pulsar_name, 'N', "name");
  arg->set_help ("set the source name");
    
  /* ***********************************************************************

  Clock/Time Options
  
  *********************************************************************** */

  menu.add ("\n" "Clock/Time options:");

  arg = menu.add (offset_clock, 'C', "offset");
  arg->set_help ("adjust clock by offset seconds");

  arg = menu.add (mjd_string, 'm', "MJD");
  arg->set_help ("set the start MJD of the observation");

  /* ***********************************************************************

  RFI Removal (SK) Options
  
  *********************************************************************** */
  menu.add ("\n" "RFI removal options:");

  arg = menu.add (config->sk_zap, "skz");
  arg->set_help ("apply spectral kurtosis filterbank RFI zapping");

  arg = menu.add (config->sk_m, "skzm", "samples");
  arg->set_help ("samples to integrate for spectral kurtosis statistics");

  arg = menu.add (config->sk_std_devs, "skzs", "stddevs");
  arg->set_help ("number of std deviations to use for spectral kurtosis excisions");

  arg = menu.add (config->sk_chan_start, "skz_start", "chan");
  arg->set_help ("first channel where signal is expected");

  arg = menu.add (config->sk_chan_end, "skz_end", "chan");
  arg->set_help ("last channel where signal is expected");

  arg = menu.add (config->sk_no_fscr, "skz_no_fscr");
  arg->set_help ("do not use SKDetector Fscrunch feature");

  arg = menu.add (config->sk_no_tscr, "skz_no_tscr");
  arg->set_help ("do not use SKDetector Tscrunch feature");

  arg = menu.add (config->sk_no_ft, "skz_no_ft");
  arg->set_help ("do not use SKDetector despeckeler");

#ifdef HAVE_CUFFT
  arg = menu.add (config->sk_nthreads, "skzn", "threads");
  arg->set_help ("number of CPU threads for spectral kurtosis filterbank");
#endif

  /* ***********************************************************************

  Dispersion removal Options
  
  *********************************************************************** */

  menu.add ("\n" "Dispersion removal options:");

  arg = menu.add (config->filterbank, 'F', "N[:D]");
  arg->set_help ("create an N-channel filterbank");
  arg->set_long_help
    ("either simply specify the number of channels; e.g. -F 256 \n"
     "or perform coherent dedispersion during the filterbank with -F 256:D \n"
     "or perform coherent dedispersion before the filterbank with -F 256:B \n"
     "or reduce the spectral leakage function bandwidth with -F 256:<N> \n"
     "where <N> is the reduction factor");

  arg = menu.add (config->plfb_nbin, 'G', "nbin");
  arg->set_help ("create phase-locked filterbank");

  arg = menu.add (config->cyclic_nchan, "cyclic", "N");
  arg->set_help ("form cyclic spectra with N channels (per input channel)");

  double dm = -1.0;
  arg = menu.add (dm, 'D', "dm");
  arg->set_help ("over-ride dispersion measure");

  arg = menu.add (config->interchan_dedispersion, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  string fft_length;
  arg = menu.add (fft_length, 'x', "nfft");
  arg->set_help ("over-ride optimal transform length");

  arg = menu.add (config->zap_rfi, 'R');
  arg->set_help ("apply time-variable narrow-band RFI filter");

  arg = menu.add (config->calibrator_database_filename, "pac", "dbase");
  arg->set_help ("pac database for phase-coherent matrix convolution");
  arg->set_long_help
    ("specify the name of a database created by pac from which to select\n"
     "the polarization calibrator to be used for matrix convolution");

  arg = menu.add (config->use_fft_bench, "fft-bench");
  arg->set_help ("use benchmark data to choose optimal FFT length");

  /* ***********************************************************************

  Detection Options
  
  *********************************************************************** */

  menu.add ("\n" "Detection options:");

  arg = menu.add (config->npol, 'd', "npol");
  arg->set_help ("1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP");

  arg = menu.add (config->ndim, 'n', "ndim");
  arg->set_help ("[experimental] ndim of output when npol=4");

  arg = menu.add (config->fourth_moment, '4');
  arg->set_help ("compute fourth-order moments");

  /* ***********************************************************************

  Folding Options
  
  *********************************************************************** */

  menu.add ("\n" "Folding options:");

  int nbin = 0;
  arg = menu.add (nbin, 'b', "nbin");
  arg->set_help ("number of phase bins in folded profile");

  arg = menu.add (config->folding_period, 'c', "period");
  arg->set_help ("folding period (in seconds)");

  arg = menu.add (config->reference_phase, 'p', "phase");
  arg->set_help ("reference phase of rising edge of bin zero");

  vector<string> ephemeris;
  arg = menu.add (ephemeris, 'E', "file");
  arg->set_help ("pulsar ephemeris used to generate predictor");

  vector<string> predictor;
  arg = menu.add (predictor, 'P', "file");
  arg->set_help ("phase predictor used for folding");

  arg = menu.add (config->additional_pulsars, 'X', "name");
  arg->set_help ("additional pulsar to be folded");

#if HAVE_CUFFT
  arg = menu.add (config->asynchronous_fold, "asynch-fold");
  arg->set_help ("fold on CPU while processing on GPU");
#endif

  /* ***********************************************************************

  Division Options
  
  *********************************************************************** */

  menu.add ("\n" "Time division options:");

  arg = menu.add (config->single_archive, 'A');
  arg->set_help ("output single archive with multiple integrations");

  arg = menu.add (config->subints_per_archive, "nsub", "N");
  arg->set_help ("output archives with N integrations each");

  arg = menu.add (config->integration_length, 'L', "seconds");
  arg->set_help ("create integrations of specified duration");

  arg = menu.add (config->single_pulse, 's');
  arg->set_help ("create single pulse integrations");

  arg = menu.add (config->fractional_pulses, 'y');
  arg->set_help ("output partially completed integrations");

  arg = menu.add (config->minimum_integration_length, "Lmin", "seconds");
  arg->set_help ("minimum integration length output");

  /* ***********************************************************************

  Output Archive Options
  
  *********************************************************************** */

  menu.add ("\n" "Output archive options:");

  arg = menu.add (config->archive_class, 'a', "archive");
  arg->set_help ("output archive class name");

  arg = menu.add (config->archive_extension, 'e', "ext");
  arg->set_help ("output filename extension");

  arg = menu.add (config->archive_filename, 'O', "name");
  arg->set_help ("output filename");

  arg = menu.add (config->pdmp_output, 'Y');
  arg->set_help ("output pdmp extras");

  vector<string> jobs;
  arg = menu.add (jobs, 'j', "job");
  arg->set_help ("psrsh command run before output");

  string script;
  arg = menu.add (script, 'J', "a.psh");
  arg->set_help ("psrsh script run before output");

  /* ***********************************************************************

  Output Archive Options
  
  *********************************************************************** */


  menu.parse (argc, argv);

  if (!config->metafile.empty())
    stringfload (&filenames, config->metafile);
  else
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    cerr << "dspsr: please specify filename[s]  (or -h for help)" << endl;
    exit (-1);
  }

  if (config->integration_length && config->minimum_integration_length < 0)
  {
    /*
      rationale: If data are divided into blocks, and blocks are 
      sent down different data reduction paths, then it is possible
      for blocks on different paths to overlap by a small amount.
      
      The minimum integration length is a simple attempt to avoid
      producing a small overlap archive with the same name as the
      full integration length archive.
  
      If minimum_integration_length is not specified, a default of 10%
      of the integration length is applied.
    */

    config->minimum_integration_length = 0.1 * config->integration_length;
  }

  // interpret the unpacker options

  for (unsigned i=0; i<unpack.size(); i++)
  {
    const char* carg = unpack[i].c_str();

    int scanned = sscanf (carg, "n%u", &config->excision_nsample);
    if (scanned == 1)
    {
      cerr << "dspsr: Using " << config->excision_nsample 
	   << " samples to estimate undigitized power" << endl;
      continue;
    }

    scanned = sscanf (carg, "c%f", &config->excision_cutoff);
    if (scanned == 1)
    {
      cerr << "dspsr: Setting impulsive interference excision threshold to "
	   << config->excision_cutoff << endl;
      continue;
    }

    scanned = sscanf (carg, "t%f", &config->excision_threshold);
    if (scanned == 1)
    {
      cerr << "dspsr: Setting two-bit sampling threshold to "
	   << config->excision_threshold << endl;
      continue;
    }
  } 

  // interpret the nbin argument
  if (nbin < 0)
  {
    config->force_sensible_nbin = true;
    config->nbin = -nbin;
  }
  if (nbin > 0)
    config->nbin = nbin;
   
  // over-ride the dispersion measure
  if (dm != -1.0)
  {
    config->dispersion_measure = dm;

    if (dm == 0.0)
    {
      cerr << "dspsr: Disabling coherent dedispersion" << endl;
      config->coherent_dedispersion = false;
    }
  }

  for (unsigned i=0; i<ephemeris.size(); i++)
  {
    cerr << "dspsr: Loading ephemeris from " << ephemeris[i] << endl;
    config->ephemerides.push_back
      ( factory<Pulsar::Parameters> (ephemeris[i]) );
  }

  for (unsigned i=0; i<predictor.size(); i++)
  {
    cerr << "dspsr: Loading phase model from " << predictor[i] << endl;
    config->predictors.push_back 
      ( factory<Pulsar::Predictor> (predictor[i]) );
  }

  for (unsigned i=0; i<jobs.size(); i++)
    separate (jobs[i], config->jobs, ",");

  if (!script.empty())
    loadlines (script, config->jobs);

  if (!pulsar_name.empty())
  {
    if (file_exists(pulsar_name.c_str()))
    {
      cerr << "dspsr: Loading source names from " << pulsar_name << endl;
      vector <string> names;
      stringfload (&names, pulsar_name);
      
      if (names.size())
	pulsar_name = names[0];
      for (unsigned i=1; i < names.size(); i++)
	config->additional_pulsars.push_back ( names[i] );
    }
    else
      cerr << "dspsr: Source name set to " << pulsar_name << endl;
  }
  
  if (!ram_min.empty())
  {
    double MB = fromstring<double> (ram_min);
    cerr << "dspsr: Using at least " << MB << " MB" << endl;
    config->set_minimum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
  }

  if (!ram_limit.empty())
  {
    if (ram_limit == "min")
      config->set_times_minimum_ndat( 1 );

    else
    {
      unsigned times = 0;
      if ( sscanf(ram_limit.c_str(), "minX%u", &times) == 1 )
	config->set_times_minimum_ndat( times );

      else
      {
	double MB = fromstring<double> (ram_limit);
	config->set_maximum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
      }
    }
  }

  if (!fft_length.empty())
  {
    char* carg = strdup( fft_length.c_str() );
    char* colon = strchr (carg, ':');
    if (colon)
    {
      *colon = '\0';
      colon++;
      if (sscanf (colon, "%d", &config->nsmear) < 1)
      {
	fprintf (stderr,
		 "Error parsing '%s' as filterbank frequency resolution\n",
		 colon);
	exit (-1);
      }
    }

    unsigned times = 0;
    
    if (string(carg) == "min")
      config->times_minimum_nfft = 1;
    else if ( sscanf(carg, "minX%u", &times) == 1 )
      config->times_minimum_nfft = times;
    else
    {
      unsigned nfft = strtol (carg, 0, 10);
      if (colon && config->nsmear >= nfft)
      {
	cerr << "dspsr -x: nfft=" << nfft
	     << " must be greater than nsmear=" << config->nsmear << endl;
	exit (-1);
      }
      config->filterbank.set_freq_res( nfft );
    }
    delete [] carg;
  }
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

