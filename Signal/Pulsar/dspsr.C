/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

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

#include "factory.h"
#include "dirutil.h"

#include <iostream>

#include <stdlib.h>
#include <errno.h>
#include <string.h>

using namespace std;

void parse_options (int argc, char** argv);

void usage ()
{
  cout << "dspsr - digital signal processing of pulsar signals\n"
    "Usage: dspsr [options] file1 [file2 ...] \n"
    "File handling options:\n"
    " -M metafile    load filenames from metafile\n"
    " -S seek        start processing at t=seek seconds\n"
    " -T total       process only t=total seconds\n"
    " -t threads     process using n=threads processor threads\n"
    " -U Mbtyes      upper limit on RAM usage in MB\n"
    " -W             no WeightedTimeSeries\n"
    " -z             run in psrdisp backward-compatibility mode\n"
    "\n"
    "Source options:\n"
    " -B bandwidth   set the bandwidth\n"
    " -f frequency   set the centre frequency\n"
    " -k telescope   set the telescope name\n"
    " -N name        set the source name\n"
    "\n"
    "Clock/Time options:\n"
    " -C offset      adjust clock by offset seconds\n"
    " -m mjd         set the start MJD of the observation\n"
    "\n"
    "Two-bit unpacking options:\n"
    " -2n<nsample>   number of samples used in estimating undigitized power\n"
    " -2c<cutoff>    cutoff threshold for impulsive interference excision\n"
    " -2t<threshold> sampling threshold at record time\n"
    "\n"
    "Filterbank options:\n"
    " -F nchan       create an nchan-channel filterbank\n"
    " -F nchan:redn  reduce spectral leakage function bandwidth by redn\n"
    " -F nchan:D     perform simultaneous coherent dedispersion\n"
    " -G nbin        create phase-locked filterbank with nbin phase bins\n"
    " -o             set psrfft up to generate optimized transforms [false]\n" 
    "\n"
    "Dedispersion/Convolution options:\n"
    " -D dm          over-ride dispersion measure\n"
    " -K             remove inter-channel dispersion delays \n"
    " -x nfft        over-ride optimal transform length\n"
    " -R             apply RFI filter in frequency domain\n"
    " -Z lib         choose the FFT library ('-Z help' for availability)\n"
    "\n"
    "Detection options:\n"
    " -d npol        1=PP+QQ, 2=PP,QQ, 3 = (PP+QQ)^2 4=PP,QQ,PQ,QP\n"
    " -n ndim        ndim of detected TimeSeries [4]\n"
    " -L nlag        form nlag ACF of the undetected data (using nlag*2 PSD)\n"
    " -L nlag:nchan  form nlag ACF of the undetected data (using nchan PSD)\n"
    " -l nlag        form lag spectrum of detected data\n"
    "\n"
    "Folding options:\n"
    " -b nbin        fold pulse profile into nbin phase bins \n"
    " -c period      fold with constant period (in seconds)\n"
    " -p phase       reference phase of pulse profile bin zero \n"
    " -E psr.eph     add the pulsar ephemeris, psr.eph, for use \n"
    " -P psr.poly    add the phase predictor, psr.poly, for use \n"
    " -X name        add another pulsar to be folded \n"
    "\n"
    "Division options:\n"
    " -A             produce a single archive with multiple Integrations \n"
    " -L seconds     form sub-integrations of the specified length \n"
    " -s             generate single pulse integrations \n"
    " -y             output partially completed single pulse integrations \n"
    "\n"
    "Output Archive options:\n"
    " -a archive     set the output archive class name\n"
    " -e ext         set the output archive filename extension\n"
    " -O filename    set the output archive filename (including extension)\n"
    " -j job[,job2]  run the psrsh commands on the output before unloading \n"
    " -J jobs.psh    run the psrsh script on the output before unloading\n"
       << endl;
}

void info ()
{
  cerr << "dspsr " << dsp::version 
       << " <" << FTransform::get_library() << ">" << endl;
}

static bool verbose = false;

// sets up the LoadToFold engine using the following attributes

void prepare (dsp::LoadToFold* engine, dsp::Input* input);

// number of time samples loaded from file at a time
uint64 block_size = 0;

// number of seconds to seek into data
double seek_seconds = 0.0;

// number of seconds to process from data
double total_seconds = 0.0;

// number of seconds to adjust clocks by
double offset_clock = 0.0;

// set the MJD
char* mjd_string = 0;

// set the telescope code
char* telescope = 0;

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
unsigned nthread = 1;

// load filenames from the ascii file named metafile
char* metafile = 0;

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFold::Config;


  vector<string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return 0;
  }

  Reference::To<dsp::LoadToFold> engine;

  if (nthread > 1)
    engine = new dsp::LoadToFoldN (nthread);
  else
    engine = new dsp::LoadToFold1;

  // configure the processing engine
  engine->set_configuration( config );

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;
    
    prepare (engine, dsp::File::create( filenames[ifile] ));
        
    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    engine->run();
    engine->finish();
  }
  catch (Error& error)
  {
    cerr << error << endl;
  }

  return 0;
}

catch (Error& error)
{
  cerr << "Error thrown: " << error << endl;
  return -1;
}


void prepare (dsp::LoadToFold* engine, dsp::Input* input)
{
  engine->set_input( input );

  dsp::Observation* info = input->get_info();

  if (info->get_detected() && !baseband_options.empty())
    throw Error (InvalidState, "prepare",
		 "input type " + input->get_name() +
		 " yields detected data and the command line option(s):"
		 "\n\n" + baseband_options + "\n\n"
		 " are specific to baseband (undetected) data.");

  if (bandwidth != 0)
  {
    cerr << "dspsr: over-riding bandwidth"
      " old=" << info->get_bandwidth() <<
      " new=" << bandwidth << endl;
    info->set_bandwidth (bandwidth);
  }
  
  if (centre_frequency != 0)
  {
    cerr << "dspsr: over-riding centre_frequency"
      " old=" << info->get_centre_frequency() <<
      " new=" << centre_frequency << endl;
    info->set_centre_frequency (centre_frequency);
  }
  
  if (telescope)
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
  
  if (mjd_string != 0)
  {
    MJD mjd (mjd_string);
    cerr << "dspsr: over-riding start time"
      " old=" << info->get_start_time() <<
      " new=" << mjd << endl;
    info->set_start_time( mjd );
  }
  
  if (seek_seconds)
    input->seek_seconds (seek_seconds);
    
  if (total_seconds)
    input->set_total_seconds (seek_seconds + total_seconds);
  
  engine->prepare ();    
}

void parse_options (int argc, char** argv)
{
  static char* args =
    "2:a:Ab:B:c:C:d:D:e:E:f:F:G:hiIjJ:k:Kl:"
    "L:m:M:n:N:O:op:P:qQrRsS:t:T:U:vVWx:X:yzZ:";

  string stropt;

  int c;
  int scanned;

  while ((c = getopt(argc, argv, args)) != -1)
  {
    if (optarg)
      stropt = optarg;

    errno = 0;

    switch (c) {

      // two-bit correction parameters
    case '2':

      baseband_options += " -2" + stropt;

      scanned = sscanf (optarg, "n%u", &config->excision_nsample);
      if (scanned == 1)  {
        cerr << "dspsr: Using " << config->excision_nsample 
             << " samples to estimate undigitized power" << endl;
        break;
      }

      scanned = sscanf (optarg, "c%f", &config->excision_cutoff);
      if (scanned == 1)  {
        cerr << "dspsr: Setting impulsive interference excision threshold "
	  "to " << config->excision_cutoff << endl;
        break;
      }

      scanned = sscanf (optarg, "t%f", &config->excision_threshold);
      if (scanned == 1) {
        cerr << "dspsr: Setting two-bit sampling threshold to "
             << config->excision_threshold << endl;
        break;
      }
      
      cerr << "dspsr: error parsing " << optarg << " as"
	" two-bit correction nsample, threshold, or cutoff" << endl;
      exit (-1);

    case 'A':
      config->single_archive = true;
      break;

    case 'a':
      config->archive_class = optarg;
      break;

    case 'B':
      bandwidth = strtod (optarg, 0);
      break;

    case 'b':
      config->nbin = strtol (optarg, 0, 10);
      break;

    case 'C':
      offset_clock = strtod (optarg, 0);
      break;

    case 'c':
      config->folding_period = strtod (optarg, 0);
      break;

    case 'D':
      config->dispersion_measure = strtod (optarg, 0);
      if (config->dispersion_measure == 0.0) {
	cerr << "dspsr: Disabling dedispersion" << endl;
	config->coherent_dedispersion = false;
      }
      else
        cerr << "dspsr: Setting DM=" << config->dispersion_measure << endl;
      break;

    case 'd':
      config->npol = strtol (optarg, 0, 10);
      break;

    case 'E':
      cerr << "dspsr: Loading ephemeris from " << optarg << endl;
      config->ephemerides.push_back ( factory<Pulsar::Parameters> (optarg) );
      break;

    case 'e':
      config->archive_extension = optarg;
      break;

    case 'F':
    {
      baseband_options += " -F" + stropt;

      char* pfr = strchr (optarg, ':');
      if (pfr)
      {
	*pfr = '\0';
	pfr++;
	if (*pfr == 'D' || *pfr == 'd')
	{
	  // FLAG that says "set the spectral resolution of the filterbank
	  // to match that required by coherent dedispersion
	  config->simultaneous_filterbank = true;
	}
	else
	{
	  if (sscanf (pfr, "%u", &config->nfft) < 1)
	  {
	    fprintf (stderr,
		     "Error parsing %s as filterbank frequency resolution\n",
		     optarg);
	    exit (-1);
	  }
	}
      }
      if (sscanf (optarg, "%u", &config->nchan) < 1)
      {
	fprintf(stderr,
		"Cannot parse '%s' as number of filterbank channels\n",
		optarg);
	exit (-1);
      }
      break;
    }
    
    case 'f':
      centre_frequency = strtod (optarg, 0);
      break;

    case 'G': {

      baseband_options += " -G" + stropt;

      char* pfr = strchr (optarg, ':');
      if (pfr) {
        *pfr = '\0';
        pfr++;
        if (sscanf (pfr, "%u", &config->plfb_nchan) < 1) {
          fprintf (stderr, "Cannot parse '%s' as "
                   "phase-locked filterbank config->nchan\n", pfr);
        exit (-1);
        }
      }
      if (sscanf (optarg, "%u", &config->plfb_nbin) < 1) {
        fprintf (stderr, "Cannot parse '%s' as "
                 "phase-locked filterbank config->nbin\n", optarg);
        exit (-1);
      }
      break;
    }

    case 'h':
      usage ();
      exit (0);

    case 'i':
      info ();
      exit (0);

#if ACTIVATE_MKL      
    case 'I':
      use_incoherent_filterbank = true;
      break;
#endif      

    case 'j':
      separate (optarg, config->jobs, ",");
      break;
      
    case 'J':
      loadlines (optarg, config->jobs);
      break;

    case 'k':
      telescope = optarg;
      break;

    case 'K':
      config->interchan_dedispersion = true;
      break;

    case 'L':
      config->integration_length = strtod (optarg, 0);
      break;

    case 'm':
      mjd_string = optarg;
      break;

    case 'M':
      metafile = optarg;
      break;

    case 'n':
      config->ndim = strtol (optarg, 0, 10);
      break;

    case 'N':
      pulsar_name = optarg;
      break;

    case 'O':
      config->archive_filename = optarg;
      break;

    case 'o':
      FTransform::optimize = true;
      break;

    case 'P':
      cerr << "dspsr: Loading phase model from " << optarg << endl;
      config->predictors.push_back ( factory<Pulsar::Predictor> (optarg) );
      break;

    case 'p':
      config->reference_phase = strtod (optarg, 0);
      cerr << "dspsr: reference phase of pulse profile bin zero = "
           << config->reference_phase << endl;
      break;

    case 'r':
      dsp::Operation::record_time = true;
      dsp::Operation::report_time = false;
      break;

    case 'R':
      baseband_options += " -R";
      config->zap_rfi = true;
      break;

    case 'S':
      seek_seconds = strtod (optarg, 0);
      break;

    case 's':
      config->single_pulse = true;
      break;

    case 'T':
      total_seconds = strtod (optarg, 0);
      break;

    case 't':
      nthread = strtol (optarg, 0, 10);
      FTransform::nthread = nthread;
      break;

    case 'U':
    {
      if (string(optarg) == "min")
      {
	config->set_times_minimum_ndat( 1 );
	break;
      }

      {
	unsigned times = 0;
	if ( sscanf(optarg, "minX%u", &times) == 1 )
	{
	  config->set_times_minimum_ndat( times );
	  break;
	}
      }

      config->set_maximum_RAM (uint64( strtod (optarg, 0) * 1024 * 1024 ));
      break;
    }

    case 'V':
      cerr << "dspsr: Entering very verbose mode" << endl;
      Pulsar::Archive::set_verbosity (3);
      dsp::set_verbosity (3);
      verbose = true;
      break;

    case 'v':
      Pulsar::Archive::set_verbosity (2);
      dsp::set_verbosity (2);
      verbose = true;
      break;

    case 'Q':
      config->report_vitals = 0;
    case 'q':
      config->report_done = 0;
      break;

    case 'W':
      config->weighted_time_series = false;
      break;

    case 'x':
    {
      char* colon = strchr (optarg, ':');
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

      if (string(optarg) == "min")
	config->times_minimum_nfft = 1;
      else if ( sscanf(optarg, "minX%u", &times) == 1 )
	config->times_minimum_nfft = times;
      else
	config->nfft = strtol (optarg, 0, 10);

      break;
    }

    case 'X':
      config->additional_pulsars.push_back (optarg);
      break;

    case 'y':
      config->fractional_pulses = true;
      break;

    case 'z':
      dsp::psrdisp_compatible = true;
      break;

    case 'Z': {

      string lib = optarg;

      if (lib == "help") {
	unsigned nlib = FTransform::get_num_libraries ();
	cerr << "dspsr: " << nlib << " available FFT libraries:";
	for (unsigned ilib=0; ilib < nlib; ilib++)
	  cerr << " " << FTransform::get_library_name (ilib);
	
	cerr << "\ndspsr: default FFT library " 
	     << FTransform::get_library() << endl;

	exit (0);
      }
      else if (lib == "simd")
        FTransform::simd = true;
      else {
	FTransform::set_library (lib);
	cerr << "dspsr: FFT library set to " << lib << endl;
      }

      break;
    }

    default:
      cerr << "invalid param '" << c << "'" << endl;
      exit (-1);

    }

    if (errno != 0)
    {
      cerr << "error parsing -" << c;
      if (optarg)
	cerr << " " << optarg;
      cerr << endl;
      perror ("");
      exit (-1);
    }
  }
}
