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

using namespace std;

static char* args =
"2:a:Ab:B:c:C:d:D:e:E:f:F:G:hiIjJ:k:Kl:L:m:M:n:N:O:op:P:RsS:t:T:U:vVWx:X:zZ:";

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
    " -k telid       set the tempo telescope code\n"
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
    " -c period      fold with constant period\n"
    " -p phase       reference phase of pulse profile bin zero \n"
    " -E psr.eph     add the pulsar ephemeris, psr.eph, for use \n"
    " -P psr.poly    add the phase predictor, psr.poly, for use \n"
    " -X name        add another pulsar to be folded \n"
    "\n"
    "Division options:\n"
    " -A             produce a single archive with multiple Integrations \n"
    " -j             join files into contiguous observation \n"
    " -L seconds     form sub-integrations of the specified length \n"
    " -s             generate single pulse Integrations \n"
    "\n"
    "Output Archive options:\n"
    " -a archive     set the output archive class name\n"
    " -e ext         set the output archive filename extension\n"
    " -O filename    set the output archive filename (including extension)\n"
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

const unsigned MB = 1024 * 1024;

// maximum number of bytes to load into RAM (default 256 MB)
uint64 maximum_RAM = 256 * MB;

// number of seconds to seek into data
double seek_seconds = 0.0;

// number of seconds to process from data
double total_seconds = 0.0;

// number of seconds to adjust clocks by
double offset_clock = 0.0;

// set the MJD
char* mjd_string = 0;

// set the telescope code
char telescope_code = 0;

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


int main (int argc, char** argv) try {

  config = new dsp::LoadToFold::Config;

  // treat all files as though they were one contiguous observation
  bool join_files = false;

  // load filenames from the ascii file named metafile
  char* metafile = 0;

  string stropt;

  int c;
  int scanned;

  while ((c = getopt(argc, argv, args)) != -1) {

    if (optarg)
      stropt = optarg;

    errno = 0;

    switch (c) {

      // two-bit correction parameters
    case '2':

      baseband_options += " -2" + stropt;

      scanned = sscanf (optarg, "n%u", &config->tbc_nsample);
      if (scanned == 1)  {
        cerr << "dspsr: Using " << config->tbc_nsample 
             << " samples to estimate undigitized power" << endl;
        break;
      }

      scanned = sscanf (optarg, "c%f", &config->tbc_cutoff);
      if (scanned == 1)  {
        cerr << "dspsr: Setting impulsive interference excision threshold "
	  "to " << config->tbc_cutoff << endl;
        break;
      }

      scanned = sscanf (optarg, "t%f", &config->tbc_threshold);
      if (scanned == 1) {
        cerr << "dspsr: Setting two-bit sampling threshold to "
             << config->tbc_threshold << endl;
        break;
      }
      
      cerr << "dspsr: error parsing " << optarg << " as"
	" two-bit correction nsample, threshold, or cutoff" << endl;
      return -1;

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

    case 'F': {

      baseband_options += " -F" + stropt;

      char* pfr = strchr (optarg, ':');
      if (pfr) {
	*pfr = '\0';
	pfr++;
	if (*pfr == 'D' || *pfr == 'd') {
	  // FLAG that says "set the spectral resolution of the filterbank
	  // to match that required by coherent dedispersion
	  config->simultaneous_filterbank = true;
	}
	else {
	  if (sscanf (pfr, "%d", &config->fres) < 1) {
	    fprintf (stderr,
		     "Error parsing %s as filterbank frequency resolution\n",
		     optarg);
	    return -1;
	  }
	}
      }
      if (sscanf (optarg, "%d", &config->nchan) < 1) {
	fprintf(stderr,
		"Cannot parse '%s' as number of filterbank channels\n",
		optarg);
	return -1;
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
        return -1;
        }
      }
      if (sscanf (optarg, "%u", &config->plfb_nbin) < 1) {
        fprintf (stderr, "Cannot parse '%s' as "
                 "phase-locked filterbank config->nbin\n", optarg);
        return -1;
      }
      break;
    }

    case 'h':
      usage ();
      return 0;

    case 'i':
      info ();
      return 0;

#if ACTIVATE_MKL      
    case 'I':
      use_incoherent_filterbank = true;
      break;
#endif      

    case 'j':
      join_files = true;
      break;

    case 'J':
      config->script = optarg;
      break;

    case 'k':
      telescope_code = optarg[0];
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
      break;

    case 'U':
      maximum_RAM = uint64( strtod (optarg, 0) * double(MB) );
      break;

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

    case 'W':
      config->weighted_time_series = false;
      break;

    case 'x': 
      config->nfft = strtol (optarg, 0, 10);
      break;

    case 'X':
      config->additional_pulsars.push_back (optarg);
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

	return 0;
      }
      else {
	FTransform::set_library (lib);
	cerr << "dspsr: FFT library set to " << lib << endl;
      }

      break;
    }

    default:
      cerr << "invalid param '" << c << "'" << endl;
      return -1;

    }

    if (errno != 0) {
      cerr << "error parsing -" << c;
      if (optarg)
	cerr << " " << optarg;
      cerr << endl;
      perror ("");
      return -1;
    }

  }

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

  if (join_files && filenames.size() == 1) {
    cerr << "Only one file specified.  Ignoring -j (join files)" << endl;
    join_files = false;
  }

  Reference::To<dsp::LoadToFold> engine;

  if (nthread > 1)
    engine = new dsp::LoadToFoldN (nthread);
  else
    engine = new dsp::LoadToFold1;

  // configure the processing engine
  engine->set_configuration( config );

  if (join_files) {

    if (verbose)
      cerr << "Opening Multfile" << endl;
    
    dsp::MultiFile* multifile = new dsp::MultiFile;
    multifile->open (filenames);

    prepare (engine, multifile);
    engine->run ();
    engine->finish ();

    return 0;

  }

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;
    
    prepare (engine, dsp::File::create( filenames[ifile] ));
        
    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    engine->run();
    engine->finish();

  }
  catch (Error& error) {
    cerr << error << endl;
  }

  return 0;
}

catch (Error& error) {
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

  if (bandwidth != 0) {
    cerr << "dspsr: over-riding bandwidth"
      " old=" << info->get_bandwidth() <<
      " new=" << bandwidth << endl;
    info->set_bandwidth (bandwidth);
  }
  
  if (centre_frequency != 0) {
    cerr << "dspsr: over-riding centre_frequency"
      " old=" << info->get_centre_frequency() <<
      " new=" << centre_frequency << endl;
    info->set_centre_frequency (centre_frequency);
  }
  
  if (telescope_code != 0) {
    cerr << "dspsr: over-riding telescope code"
      " old=" << info->get_telescope_code() <<
      " new=" << telescope_code << endl;
    info->set_telescope_code (telescope_code);
  }
  
  if (!pulsar_name.empty()) {
    cerr << "dspsr: over-riding source name"
      " old=" << info->get_source() <<
      " new=" << pulsar_name << endl;
    info->set_source( pulsar_name );   
  }
  
  if (mjd_string != 0) {
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
  
  uint64 this_block_size = block_size;
  
  if (!this_block_size) {
    
    /*
      This simple calculation of the maximum block size does not
      consider the RAM required for out of place operations, FFT
      plans, etc.
    */

    unsigned nbit  = info->get_nbit();
    unsigned ndim  = info->get_ndim();
    unsigned npol  = info->get_npol();
    unsigned nchan = info->get_nchan();
    unsigned res   = input->get_resolution();
    
    // Any outofplace operation will double the size requirements
    double copies = config->get_nbuffers();

    // each nbit number will be unpacked into a float
    double nbyte = double(nbit)/8 + copies * sizeof(float);
    
    double nbyte_dat = nbyte * ndim * npol * nchan;

    if (maximum_RAM == 0) {
  
      uint64 min = engine->get_minimum_samples();
      double inMB = double(min) * nbyte_dat / double(MB);
      cerr << "dspsr: using minimum blocksize = " << min << " samples" << endl
           << "       (equivalent to -U " << inMB << ")" << endl;
  
      input->set_block_size( min );
  
      return;
    }


    this_block_size = (uint64(maximum_RAM / nbyte_dat) / res) * res;
    
    cerr << "dspsr: block size=" << this_block_size << " samples" << endl;
    
    if (this_block_size == 0)
      throw Error (InvalidState, "dspsr:prepare",
		   "insufficient RAM: limit=%f MB  require=%f MB\n"
		   "(use -U to increase RAM limit)",
		   double(maximum_RAM)/double(MB),
		   double(nbyte_dat)/double(MB));

  }

  input->set_block_size ( this_block_size );
    
}
