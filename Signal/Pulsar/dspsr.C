#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "dsp/IOManager.h"
#include "dsp/MultiFile.h"

#include "dsp/Unpacker.h"
#include "dsp/BitSeries.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/Dedispersion.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"
#include "dsp/SubFold.h"
#include "dsp/PhaseSeries.h"

#if ACTIVATE_MPI
#include "dsp/MPIRoot.h"
#endif

#if ACTIVATE_MKL
#include "dsp/IncoherentFilterbank.h"
#endif

#if ACTIVATE_BLITZ
#include "dsp/PolyPhaseFilterbank.h"
#endif

#include "dsp/Archiver.h"

#include "Pulsar/Archive.h"

#include "fftm.h"
#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"
#include "MakeInfo.h"

static char* args = "2:a:Ab:d:D:e:E:f:F:hiIjM:n:N:op:P:sS:t:T:vVx:";

void usage ()
{
  cout << "dspsr - test baseband/dsp pulsar processing\n"
    "Usage: dspsr [" << args << "] file1 [file2 ...] \n"
    "File handling options:\n"
    " -a archive     set the output archive class name\n"
    " -A             produce a single archive with multiple Integrations\n"
    " -e ext         set the output archive filename extension\n"
    " -E filename    set the output archive filename (including extension)\n"
    " -j             join files into contiguous observation\n"
    " -M metafile    load filenames from metafile\n"
    " -N name        Over-ride name of pulsar with this name\n"
    " -s             generate single pulse Integrations\n"
    " -S seek        Start processing at t=seek seconds\n"
    " -t gulps       Stop processing after this many gulps\n"
    " -T total       Process only t=total seconds\n"
    "\n"
    "Two-bit unpacking options:\n"
    " -2n<nsample>   number of samples used in estimating undigitized power\n"
    " -2c<cutoff>    cutoff threshold for impulsive interference excision\n"
    " -2t<threshold> sampling threshold at record time\n"
    "\n"
    "Filterbank options:\n"
    " -f ffts        Perform this many forward FFTs per gulp [16]\n"
    " -F nchan       create an nchan-channel filterbank\n"
    " -F nchan:redn  reduce spectral leakage function bandwidth by redn\n"
    " -F nchan:D     perform simultaneous coherent dedispersion\n"
#if ACTIVATE_MKL
    " -I             Over-ride with IncoherentFilterbank class [false]\n"
#endif
    " -o             Set psrfft up to generate optimized transforms [false]\n" 
    "\n"
    "Dedispersion/Convolution options:\n"
    " -D dm          over-ride dispersion measure\n"
    " -x nfft        over-ride optimal transform length\n"
    "\n"
    "Detection options:\n"
    " -d npol        1=PP+QQ, 2=PP,QQ, 4=PP,QQ,PQ,QP\n"
    " -n ndim        ndim of detected TimeSeries [4]\n"
    "\n"
    "Folding options:\n"
    " -b nbin        fold pulse profile into nbin phase bins \n"
    " -p phase       reference phase of pulse profile bin zero \n"
    " -P psr.eph     add the pulsar ephemeris, psr.eph, for use \n"
    "\n"
       << endl;
}

void info ()
{
#ifdef ACTIVATE_MPI
  char* mpimsg = ":MPI";
#else
  char* mpimsg = "";
#endif
  
  cerr << "dspsr " << dsp::version << " <" << fft::id << mpimsg 
       << "> compiled by " << MakeInfo_user << " on " << MakeInfo_date
       << endl;
}

int main (int argc, char** argv) 
{ try {

  // rank of the node on which this processor is running
  int mpi_rank = 0;
  // number of nodes in the processing group
  int mpi_size = 1;
  // rank of the root node (the one that loads from file)
  int mpi_root = 0;

#if ACTIVATE_MPI

  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

  int  name_length;
  char mpi_name [MPI_MAX_PROCESSOR_NAME];

  MPI_Get_processor_name (mpi_name, &name_length);

  // MPI error return variables
  // int  mpi_err;
  // char mpi_errstr [MPI_MAX_ERROR_STRING];

  MPI_Errhandler_set (MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#endif

  bool verbose = false;

  // number of time samples loaded from file at a time
  int blocks = 0;
  int ndim = 4;
  int nchan = 1;
  int npol = 4;
  int set_nfft = 0;

  int nbin = 0;

  // the pulse phase of profile bin zero
  double reference_phase = 0.0;

  // the ephemerides from which to choose when creating a folding polyco
  vector< psrephem* > ephemerides;

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

  // filename extension of the output archives
  string archive_extension;

  // filename of the output archives
  string archive_filename;

  // form single pulse archives
  bool single_pulse = false;

  // form a single archive with multiple integrations
  bool single_archive = false;

  // treat all files as though they were one contiguous observation
  bool join_files = false;

  // load filenames from the ascii file named metafile
  char* metafile = 0;

  // number of seconds to seek into data
  double seek_seconds = 0.0;

  // number of seconds to process from data
  double total_seconds = 0.0;

#if ACTIVATE_MKL
  // If true, a dsp::IncoherentFilterbank is used rather than a dsp::Filterbank
  bool use_incoherent_filterbank = false;
#endif

  // dispersion measure
  double dispersion_measure = 0.0;
  bool dm_set = false;

  // Pulsar name
  string pulsar_name;

  // Filename of polyphase filterbank coefficients
  char* polyphase_filter = 0;

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

    case 'A':
      single_archive = true;
      break;

    case 'a':
      archive_class = optarg;
      break;

    case 'b':
      nbin = atoi (optarg);
      break;

    case 'D':
      dispersion_measure = atof(optarg);
      dm_set = true;
      break;

    case 'd':
      npol = atoi (optarg);
      break;

    case 'E':
      archive_filename = optarg;
      break;

    case 'e':
      archive_extension = optarg;
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
      if (mpi_rank == 0)
	usage ();
      return 0;

    case 'i':
      if (mpi_rank == 0)
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

    case 'M':
      metafile = optarg;
      break;

    case 'n':
      ndim = atoi (optarg);
      break;

    case 'N':
      pulsar_name = optarg;
      break;

    case 'o':
      fft::plans.optimize = true;
      break;

    case 'P':
      ephemerides.push_back ( new psrephem (optarg) );
      break;

    case 'p':
      scanned = sscanf (optarg, "%lf", &reference_phase);
      if (scanned != 1) {
        cerr << "dspsr: Error parsing " << optarg << " as reference phase"
	     << endl;
	return -1;
      }
      cerr << "dspsr: reference phase of pulse profile bin zero = "
           << reference_phase << endl;
      break;

    case 'r':
      scanned = sscanf (optarg, "%d", &mpi_root);
      if (scanned != 1) {
        cerr << "dspsr: Error parsing " << optarg << " as MPI root node"
	     << endl;
	return -1;
      }
      break;
      if (mpi_root < 0 || mpi_root >= mpi_size) {
	cerr << "dspsr: Invalid MPI root node = " << mpi_root << endl;
	return -1;
      }

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

    case 'x': 
      set_nfft = atoi (optarg);
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
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

  if (nchan != 1 && polyphase_filter) {
    cerr << "dsp: cannot use both Filterbank (-F) and PolyPhaseFilterbank (-P)"
	 << endl;
    return -1;
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

#if ACTIVATE_MPI

  dsp::MPIRoot* mpi_data = 0;

  if (mpi_size > 1) {

    if (verbose)
      cerr << "Creating MPIRoot instance" << endl;
    
    mpi_data = new dsp::MPIRoot (MPI_COMM_WORLD);
    mpi_data -> set_root (mpi_root);

  }    

#endif

  if (verbose)
    cerr << "Creating Dedispersion instance" << endl;
  dsp::Dedispersion* kernel = new dsp::Dedispersion;

  if (set_nfft)
    kernel->set_frequency_resolution (set_nfft);

  if (verbose)
    cerr << "Creating Response (passband) instance" << endl;
  dsp::Response* passband = new dsp::Response;

  if (fres)
    kernel->set_frequency_resolution (fres);

  dsp::TimeSeries* convolve = voltages;

  bool need_to_detect = true;

  if (nchan > 1) {

    // output filterbank data
    convolve = new dsp::WeightedTimeSeries;

#if ACTIVATE_MKL
    if( use_incoherent_filterbank ) {

      if (verbose)
	cerr << "Creating IncoherentFilterbank instance" << endl;

      // software filterbank constructor
      dsp::IncoherentFilterbank* filterbank = new dsp::IncoherentFilterbank;
      filterbank->set_input (voltages);
      filterbank->set_output (convolve);
      filterbank->set_nchan (nchan);

      if( npol==1 ){
	filterbank->set_output_state( Signal::Intensity );
	need_to_detect = false;
      }
      else if( npol==2 ){
	filterbank->set_output_state( Signal::PPQQ );
	need_to_detect = false;
      }
      else if( npol==4 )
	filterbank->set_output_state( Signal::Analytic );
      else {
	cerr << "dspsr: invalid npol=" << npol << endl;
	return -1;
      }

      operations.push_back( filterbank );
    }
    else
#endif
	
      {  
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
    
  }
  
#if ACTIVATE_BLITZ

  if (polyphase_filter) {
	
    if (verbose)
      cerr << "Creating PolyPhaseFilterbank instance" << endl;
    
    // polyphase filterbank constructor
    dsp::PolyPhaseFilterbank* filterbank = new dsp::PolyPhaseFilterbank;
    
    filterbank->set_input (voltages);
    filterbank->set_output (convolve);

    filterbank->load_complex_coefficients (polyphase_filter);
    nchan = filterbank->get_nchan();

    cerr << "dspsr: PolyPhaseFilterbank nchan=" << nchan << " loaded" << endl;

    operations.push_back (filterbank);
    
  }

#endif

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
  
  if( need_to_detect ) {
    
    if (verbose)
      cerr << "Creating Detection instance" << endl;
    dsp::Detection* detect = new dsp::Detection;
    
    if (npol == 4) {
      detect->set_output_state (Signal::Coherence);
      detect->set_output_ndim (ndim);
    }
    else if (npol == 2)
      detect->set_output_state (Signal::PPQQ);
    else if (npol == 1)
      detect->set_output_state (Signal::Intensity);
    else {
      cerr << "dspsr: invalid npol=" << npol << endl;
      return -1;
    } 

    detect->set_input (convolve);
    detect->set_output (convolve);
    
    operations.push_back (detect);
  }

  if (verbose)
    cerr << "Creating Archiver instance" << endl;
  dsp::Archiver* archiver = new dsp::Archiver;

  cerr << "dspsr: Archive class name = " << archive_class << endl;

  archiver->set_archive_class (archive_class.c_str());

  if (!archive_extension.empty())
    archiver->set_extension (archive_extension);

  if (!archive_filename.empty()) {
    archiver->set_filename (archive_filename);

    if (!archive_extension.empty())
      cerr << "dspsr: Warning: archive extension will be ignored" << endl;
  }

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

  for (unsigned ieph=0; ieph < ephemerides.size(); ieph++)
    fold->add_pulsar_ephemeris ( ephemerides[ieph] );

  fold->set_input (convolve);
  fold->set_output (profiles);

  operations.push_back (fold);

  dsp::Operation::record_time = true;

  if (join_files) {

    if (mpi_rank == mpi_root) {

      if (verbose)
	cerr << "Opening Multfile" << endl;

      dsp::MultiFile* multifile = new dsp::MultiFile;
      multifile->open (filenames);
      manager->set_input (multifile);

    }

    // kludge to make the following loop operate only once
    filenames.resize(1);

  }

  // single archive instance used when single_pulse && single_archive
  Reference::To<Pulsar::Archive> archive;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (!join_files && mpi_rank == mpi_root) {

      if (verbose)
	cerr << "opening data file " << filenames[ifile] << endl;
      
      manager->open (filenames[ifile]);
        
      if (verbose)
	cerr << "data file " << filenames[ifile] << " opened" << endl;
      
    }

    // Make sure the source name used to construct kernel is set correctly
    if( pulsar_name!=string() )
      manager->get_info()->set_source( pulsar_name );   

    if (single_pulse && single_archive) {

      cerr << "Creating single pulse single archive" << endl;
      archive = Pulsar::Archive::new_Archive (archive_class);
      archiver->set_archive (archive);

    }

#if ACTIVATE_MPI

    if (mpi_size > 1 && mpi_rank != mpi_root) {

      // the processing nodes must wait here for the root node to
      // inform them of the buffer size
      
      mpi_data -> prepare ();
      manager->set_input (mpi_data);

    }    
    
#endif 

    fold->prepare ( manager->get_info() );

    double dm = fold->get_pulsar_ephemeris() -> get_dm();

    if (dm_set) {
      cerr << "dspsr: over-riding DM=" << dm << " with DM=" 
	   << dispersion_measure << endl;
      dm = dispersion_measure;
    }

    unsigned nblocks_tot = 0;
    unsigned block_size = ffts;
    unsigned overlap = 0;
    unsigned nfft = 0;
    unsigned nfilt = 0;

    vector<dsp::Operation*> active_operations;

    if (manager->get_info()->get_detected()) {

      active_operations.push_back (manager);
      active_operations.push_back (fold);

    }

    else {

      active_operations = operations;

      kernel->set_dispersion_measure (dm);
      kernel->match ( manager->get_info(), nchan);
      
      nfft = kernel->get_ndat();
      nfilt = kernel->get_impulse_pos() + kernel->get_impulse_neg();
      
      unsigned real_complex = 2 / manager->get_info()->get_ndim();
      
      block_size = ((nfft-nfilt) * ffts + nfilt) * nchan * real_complex;
      overlap = nfilt * nchan * real_complex;
      
      unsigned fft_size = nfft * nchan * real_complex;

      cerr << "FFTsize=" << fft_size << endl;
      cerr << "Blocksz=" << block_size << endl;
      cerr << "Overlap=" << overlap << endl;
      
    }

    archiver->set_operations (active_operations);

    if (mpi_rank == mpi_root) {

      if (seek_seconds)
	manager->seek_seconds (seek_seconds);
      
      if (total_seconds)
	manager->set_total_seconds (seek_seconds + total_seconds);

      manager->set_block_size ( block_size );
      manager->set_overlap ( overlap );

      unsigned ndat_good = block_size - overlap;
      nblocks_tot = manager->get_total_samples() / ndat_good;
      if (manager->get_total_samples() % ndat_good)
	nblocks_tot ++;

      cerr << "processing ";
      if (blocks)
	cerr << blocks << " out of ";
      cerr << nblocks_tot << " blocks of " << block_size << " time samples\n";
      if (blocks)
	nblocks_tot = blocks;
      
      if (nfft) {

	fprintf (stderr, "(nfft:%d  ngood:%d  ffts/job:%d  jobs:%d)\n",
		 nfft, nfft-nfilt, ffts, nblocks_tot);
	if (nchan > 1)
	  fprintf (stderr,
		   "(nchan:%d -- Resolution:: spectral:%d  temporal:%d)\n",
		   nchan, fres, 1);
	
      }

#if ACTIVATE_MPI

      if (mpi_size > 1) {
    
	mpi_data -> copy (manager);
	mpi_data -> prepare ();

	mpi_data -> set_Input ( manager->get_input() );
	mpi_data -> serve ();

	return 0;
      }    
    
#endif   

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

    profiles->zero();

    int block=0;
    int last_percent = -1;

    while (!manager->eod()) {

      for (unsigned iop=0; iop < active_operations.size(); iop++) try {

        if (verbose) cerr << "dspsr: calling " 
                          << active_operations[iop]->get_name() << endl;

        active_operations[iop]->operate ();

        if (verbose) cerr << "dspsr: " << active_operations[iop]->get_name() 
                          << " done" << endl;

      }
      catch (Error& error)  {

        cerr << error << endl;
        cerr << "dspsr: removing " << active_operations[iop]->get_name() 
             << " from operations" << endl;

        active_operations.erase (active_operations.begin()+iop);
        archiver->set_operations (operations);

        iop --;

        cerr << "dspsr: continuing with operation " << iop+1 << endl;
      }

      block++;

      if (mpi_rank == mpi_root) {

	int percent = int (100.0*float(block)/float(nblocks_tot));
	
	if (percent > last_percent) {
	  cerr << "Finished " << percent << "%\r";
	  last_percent = percent;
	}
	
	if (blocks && block==blocks)
	  break;

      }

    }

    if (verbose)
      cerr << "end of data" << endl;

    fprintf (stderr, "%25s %25s\n", "Operation", "Time Spent");
    for (unsigned iop=0; iop < operations.size(); iop++)
      fprintf (stderr, "%25s %25.2g\n", operations[iop]->get_name().c_str(),
	       (float) operations[iop]->get_total_time());

    if (!single_pulse) {

      if (verbose)
	cerr << "Creating archive" << endl;
      archiver->set_profiles (profiles);
      archiver->unload ();

    }
    else if (archive) {

      cerr << "Unloading single archive with " << archive->get_nsubint ()
           << " integrations" << endl
           << "Filename = '" << archive->get_filename() << "'" << endl;

      // archive->pscrunch ();
      // archive->fscrunch ();

      archive->unload ();

    }
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
