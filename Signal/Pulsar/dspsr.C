#include <iostream>
#include <unistd.h>

#include "dsp/IOManager.h"
#include "dsp/MultiFile.h"

#include "dsp/Unpacker.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/Dedispersion.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"
#include "dsp/SubFold.h"
#include "dsp/PhaseSeries.h"

#include "dsp/Archiver.h"

#include "Pulsar/Archive.h"

#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

static char* args = "2:a:b:f:F:hjM:n:N:p:PsS:t:T:vV";

void usage ()
{
  cout << "dspsr - test baseband/dsp pulsar processing\n"
    "Usage: dspsr [" << args << "] file1 [file2 ...] \n"
    "File handling options:\n"
    " -a archive     set the output archive class name\n"
    " -j             join files into contiguous observation\n"
    " -M metafile    load filenames from metafile\n"
    " -s             generate an archive for each single pulse\n"
    " -S seek        Start processing at t=seek seconds\n"
    " -T total       Process only t=total seconds\n"
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
    "\n"
    "Folding options:\n"
    " -b nbin        fold pulse profile into nbin phase bins\n"
    " -p phase       reference phase of pulse profile bin zero\n"
    "\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  bool verbose = false;

  // number of time samples loaded from file at a time
  int blocks = 0;
  int ndim = 4;
  int nchan = 1;

  int nbin = 0;
  double reference_phase = 0.0;

  int ffts = 16;
  int fres = 0;

  // perform coherent dedispersion during filterbank construction
  bool simultaneous = false;

  // number of time samples used to estimate undigitized power
  unsigned tbc_nsample = 0;

  // cutoff power used for impulsive interference rejection
  float tbc_cutoff = 0.0;

  // sampling threshold
  float tbc_threshold = 0.0;

  // class name of the archives to be produced
  string archive_class = "TimerArchive";

  // form single pulse archives
  bool single_pulse = false;

  // treat all files as though they were one contiguous observation
  bool join_files = false;

  // load filenames from the ascii file named metafile
  char* metafile = 0;

  // number of seconds to seek into data
  double seek_seconds = 0.0;

  // number of seconds to process from data
  double total_seconds = 0.0;

  int c;
  int scanned;

  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

      // two-bit correction parameters
    case '2':

      scanned = sscanf (optarg, "n%u", &tbc_nsample);
      if (scanned == 1)  {
        cerr << "dspsr: Using " << tbc_nsample 
             << " samples to estimate undigitized power" << endl;
        break;
      }

      scanned = sscanf (optarg, "c%f", &tbc_cutoff);
      if (scanned == 1)  {
        cerr << "dspsr: Setting impulsive interference excision threshold "
                "to " << tbc_cutoff << endl;
        break;
      }

      scanned = sscanf (optarg, "t%f", &tbc_threshold);
      if (scanned == 1) {
        cerr << "dspsr: Setting two-bit sampling threshold to "
             << tbc_threshold << endl;
        break;
      }

      cerr << "dspsr: error parsing " << optarg << " as"
	  " two-bit correction nsample, threshold, or cutoff" << endl;
      return -1;

    case 'a':
      archive_class = optarg;
      break;

    case 'b':
      nbin = atoi (optarg);
      break;

    case 'F': {
      char* pfr = strchr (optarg, ':');
      if (pfr) {
	*pfr = '\0';
	pfr++;
	if (*pfr == 'D' || *pfr == 'd') {
	  // FLAG that says "set the spectral resolution of the filterbank
	  // to match that required by coherent dedispersion
	  simultaneous = true;
	}
	else {
	  if (sscanf (pfr, "%d", &fres) < 1) {
	    fprintf (stderr,
		     "Error parsing %s as filterbank frequency resolution\n",
		     optarg);
	    return -1;
	  }
	}
      }
      if (sscanf (optarg, "%d", &nchan) < 1) {
	fprintf(stderr,
		"Cannot parse '%s' as number of filterbank channels\n",
		optarg);
	return -1;
      }
      break;
    }

    case 'f':
      ffts = atoi (optarg);
      break;

    case 'h':
      usage ();
      return 0;

    case 'j':
      join_files = true;
      break;

    case 'M':
      metafile = optarg;
      break;

    case 'n':
      ndim = atoi (optarg);
      break;

    case 'P':
      dsp::psrdisp_compatible = true;
      break;

    case 'p':
      scanned = sscanf (optarg, "%lf", &reference_phase);
      if (scanned != 1) {
        cerr << "dspsr: Error parsing " << optarg << " as reference phase"
	     << endl;
	return -1;
      }
      break;

    case 'S':
      scanned = sscanf (optarg, "%lf", &seek_seconds);
      if (scanned != 1) {
        cerr << "dspsr: Error parsing " << optarg << " as seek time" << endl;
	return -1;
      }
      break;

    case 's':
      single_pulse = true;
      break;

    case 'T':
      scanned = sscanf (optarg, "%lf", &total_seconds);
      if (scanned != 1) {
        cerr << "dspsr: Error parsing " << optarg << " as total time" << endl;
	return -1;
      }
      break;

    case 't':
      blocks = atoi (optarg);
      break;

    case 'V':
      Pulsar::Archive::set_verbosity (3);
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      dsp::Shape::verbose = true;
    case 'v':
      dsp::Archiver::verbose = true;
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

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

  if (verbose)
    cerr << "Creating WeightedTimeSeries instance" << endl;
  dsp::TimeSeries* voltages = new dsp::WeightedTimeSeries;

  if (verbose)
    cerr << "Creating PhaseSeries instance" << endl;
  dsp::PhaseSeries* profiles = new dsp::PhaseSeries;

  vector<dsp::Operation*> operations;

  if (verbose)
    cerr << "Creating IOManager instance" << endl;

  dsp::IOManager* manager = new dsp::IOManager;

  manager->set_output (voltages);

  operations.push_back (manager);

  if (verbose)
    cerr << "Creating Dedispersion instance" << endl;
  dsp::Dedispersion* kernel = new dsp::Dedispersion;

  if (verbose)
    cerr << "Creating Response (passband) instance" << endl;
  dsp::Response* passband = new dsp::Response;

  if (fres)
    kernel->set_frequency_resolution (fres);

  dsp::TimeSeries* convolve = voltages;

  if (nchan > 1) {

    // output filterbank data
    convolve = new dsp::WeightedTimeSeries;

    if (verbose)
      cerr << "Creating Filterbank instance" << endl;

    // software filterbank constructor
    dsp::Filterbank* filterbank = new dsp::Filterbank;
    filterbank->set_input (voltages);
    filterbank->set_output (convolve);
    filterbank->set_nchan (nchan);

    if (simultaneous) {
      filterbank->set_response (kernel);
      filterbank->set_passband (passband);
    }

    operations.push_back (filterbank);
  }

  if (nchan == 1 || !simultaneous) {

    if (verbose)
      cerr << "Creating Convolution instance" << endl;

    dsp::Convolution* convolution = new dsp::Convolution;

    convolution->set_response (kernel);
    convolution->set_passband (passband);

    convolution->set_input  (convolve);  
    convolution->set_output (convolve);  // inplace

    operations.push_back (convolution);
  }

  if (verbose)
    cerr << "Creating Detection instance" << endl;
  dsp::Detection* detect = new dsp::Detection;

  detect->set_output_state (Signal::Coherence);
  detect->set_output_ndim (ndim);
  detect->set_input (convolve);
  detect->set_output (convolve);

  operations.push_back (detect);

  if (verbose)
    cerr << "Creating Archiver instance" << endl;
  dsp::Archiver* archiver = new dsp::Archiver;

  if (verbose)
    cerr << "Setting Archiver Archive class name to " << archive_class << endl;

  archiver->set_archive_class (archive_class.c_str());

  if (verbose)
    cerr << "Creating Fold instance" << endl;
  dsp::Fold* fold;

  if (single_pulse) {
    dsp::SubFold* subfold = new dsp::SubFold;
    subfold -> set_subint_turns (1);
    subfold -> set_unloader (archiver);

    fold = subfold;
  }
  else
    fold = new dsp::Fold;

  if (nbin)
    fold->set_nbin (nbin);

  if (reference_phase)
    fold->set_reference_phase (reference_phase);

  fold->set_input (convolve);
  fold->set_output (profiles);

  operations.push_back (fold);

  dsp::Operation::record_time = true;

  if (join_files) {

      if (verbose)
	  cerr << "Opening Multfile" << endl;
      dsp::MultiFile* multifile = new dsp::MultiFile;
      multifile->open (filenames);

      manager->set_input (multifile);

      // kludge to make the following loop operate only once
      filenames.resize(1);
  }

  archiver->set_operations (operations);

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (!join_files) {
      if (verbose)
        cerr << "opening data file " << filenames[ifile] << endl;

      manager->open (filenames[ifile]);

      if (verbose)
        cerr << "data file " << filenames[ifile] << " opened" << endl;
    }

    // In the case of unpacking two-bit data, set the corresponding parameters

    dsp::TwoBitCorrection* tbc;
    tbc = dynamic_cast<dsp::TwoBitCorrection*> ( manager->get_unpacker() );

    if ( tbc && tbc_nsample )
      tbc -> set_nsample ( tbc_nsample );

    if ( tbc && tbc_threshold )
      tbc -> set_threshold ( tbc_threshold );

    if ( tbc && tbc_cutoff )
      tbc -> set_cutoff_sigma ( tbc_cutoff );



    fold->prepare ( manager->get_info() );

    double dm = fold->get_pulsar_ephemeris() -> get_dm();

    kernel->set_dispersion_measure (dm);
    kernel->match ( manager->get_info(), nchan);

    unsigned nfft = kernel->get_ndat();
    unsigned nfilt = kernel->get_impulse_pos() + kernel->get_impulse_neg();

    unsigned real_complex = 2 / manager->get_info()->get_ndim();

    unsigned block_size = ((nfft-nfilt) * ffts + nfilt) * nchan * real_complex;
    unsigned overlap = nfilt * nchan * real_complex;

    unsigned fft_size = nfft * nchan * real_complex;

    if (dsp::psrdisp_compatible && tbc) {

      unsigned ppwt = tbc->get_nsample();
      unsigned extra = ppwt - (block_size % ppwt);

      cerr << "dspsr: psrdisp compatibilty\n"
	"   adding " << extra << " samples to make block_size a multiple of " 
	   << ppwt << endl;

      block_size += extra;
      overlap += extra;

    }

    cerr << "FFTsize=" << fft_size << endl;
    cerr << "Blocksz=" << block_size << endl;
    cerr << "Overlap=" << overlap << endl;

    if (seek_seconds)
      manager->seek_seconds (seek_seconds);

    if (total_seconds)
      manager->set_total_seconds (seek_seconds + total_seconds);

    manager->set_block_size ( block_size );
    manager->set_overlap ( overlap );

    unsigned ndat_good = block_size - overlap;
    unsigned nblocks_tot = manager->get_total_samples() / ndat_good;
    if (manager->get_total_samples() % ndat_good)
      nblocks_tot ++;

    cerr << "processing ";
    if (blocks)
      cerr << blocks << " out of ";
    cerr << nblocks_tot << " blocks of " << block_size << " time samples\n";
    if (blocks)
      nblocks_tot = blocks;

    fprintf (stderr, "(nfft:%d  ngood:%d  ffts/job:%d  jobs:%d)\n",
             nfft, nfft-nfilt, ffts, nblocks_tot);
    if (nchan > 1)
    fprintf (stderr, "(nchan:%d -- Resolution:: spectral:%d  temporal:%d)\n",
             nchan, fres, 1);

    int block=0;
    int last_percent = -1;
    while (!manager->eod()) {

      try {
        for (unsigned iop=0; iop < operations.size(); iop++)
	  operations[iop]->operate ();
      }
      catch (Error& error)  {
        cerr << error << endl;
        cerr << "dspsr: Exiting data reduction loop." << endl;
        break;
      }


      block++;

      int percent = int (100.0*float(block)/float(nblocks_tot));

      if (percent > last_percent) {
        cerr << "Finished " << percent << "%\r";
        last_percent = percent;
      }

      if (blocks && block==blocks)
	break;

    }

    if (verbose)
	cerr << "end of data" << endl;

    fprintf (stderr, "%25s %25s\n", "Operation", "Time Spent");
    for (unsigned iop=0; iop < operations.size(); iop++)
      fprintf (stderr, "%25s %25.2g\n", operations[iop]->get_name().c_str(),
	       (float) operations[iop]->get_total_time());

    if (verbose)
      cerr << "Creating archive" << endl;
    archiver->set_profiles (profiles);
    archiver->unload ();

  }
  catch (string& error) {
    cerr << error << endl;
  }

  
  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "unknown exception thrown." << endl;
  return -1;
}

}
