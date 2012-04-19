/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SigProcObservation.h"
#include "dsp/SigProcDigitizer.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"

#include "dsp/Rescale.h"
#include "dsp/PScrunch.h"
#include "dsp/FScrunch.h"
#include "dsp/TScrunch.h"

#include "dirutil.h"
#include "Error.h"

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define SIGPROC_FILTERBANK_RINGBUFFER
#ifdef SIGPROC_FILTERBANK_RINGBUFFER
#include <dada_hdu.h>
#include <ipcio.h>
#include "dsp/ASCIIObservation.h"
#endif

#ifdef HAVE_OPENSSL_SHA_H
#include <openssl/sha.h>
char get_SHA_hash(unsigned char* buffer,int size, char* hashStr);
#endif
#ifdef HAVE_PSRXML
#include <psrxml.h>
#endif



using namespace std;

static char* args = "b:B:cI:o:prt:f:hxk:vV";

void usage ()
{
  cout << "the_decimator - convert dspsr input to sigproc filterbank \n"
    "Usage: the_decimator file1 [file2 ...] \n"
    "Options:\n"
    "\n"
    "  -b bits   number of bits per sample output to file \n" 
    "  -B secs   number of seconds per block \n"
    "  -c        keep offset and scale constant \n"
    "  -I secs   number of seconds between level updates \n"
    "  -o file   file stamp for filterbank file  \n" 
    "  -r        report total Operation times \n"
    "  -p        revert to FPT order \n"
    "  -t factor tscrunch by factor  \n"
    "  -f factor fscrunch by factor  \n"
#ifdef SIGPROC_FILTERBANK_RINGBUFFER
    "  -k key    shared memory key to output DADA ring buffer \n"
#endif
#ifdef HAVE_PSRXML
    "  -x        write PsrXML header file\n"
#endif
    "\n"
    "Optional Components:\n"
    "Write to DADA ring buffer "
#ifdef SIGPROC_FILTERBANK_RINGBUFFER
    "ENABLED\n"
#else
    "DISABLED (compile time switch)\n"
#endif
    "PsrXML header writing "
#ifdef HAVE_PSRXML
    "ENABLED\n"
#else
    "DISABLED (compile time switch)\n"
#endif
    "SHA block checksums "
#ifdef HAVE_OPENSSL_SHA_H
    "ENABLED\n"
#else
    "DISABLED (compile time switch)\n"
#endif
       << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;
  bool constant_offset_scale = false;
  bool write_psrxml = false;
  int nbits = 8;

#ifdef SIGPROC_FILTERBANK_RINGBUFFER
  dada_hdu_t* hdu = 0;
  key_t hdu_key = 0;
#endif

  // block size in seconds
  double block_size = 10;

  // update interval in seconds;
  double update_interval = 0.0;

  unsigned int tscrunch_factor=0;
  unsigned int fscrunch_factor=0;

  FILE* outfile = stdout;
  char* outfile_basename = 0;

  dsp::TimeSeries::Order order = dsp::TimeSeries::OrderTFP;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'b':
      nbits = atoi (optarg);
      break;

    case 'B':
      block_size = atof (optarg);
      break;

    case 'c':
      constant_offset_scale = true;
      break;

    case 'I':
      update_interval = atof (optarg);
      break;

    case 'o':
      outfile_basename = optarg;
      break;

    case 'p':
      order = dsp::TimeSeries::OrderFPT;
      break;

    case 'r':
      dsp::Operation::record_time = true;
      break;

    case 'f':
      fscrunch_factor=atoi(optarg);
      break;

    case 't':
      tscrunch_factor=atoi(optarg);
      break;


    case 'k':
      if (sscanf (optarg, "%x", &hdu_key) != 1)
      {
	cerr << "the_decimator: could not scan key from "
	  "'" << hdu_key << "'" << endl;
	return -1;
      }
      break;
    case 'x':
#ifdef HAVE_PSRXML
      write_psrxml = true;
#else
      fprintf(stderr,"PsrXML library was not enabled at compile time\n");
      exit(1);
#endif
      break;
    case 'h':
      usage ();
      return 0;
    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  if (outfile_basename)
  {
    string filename = outfile_basename;
    filename += ".fil";

    outfile = fopen (filename.c_str(),"wb"); 
    if (!outfile)
      throw Error (FailedSys, "",
		   "Could not open " + filename + " for output");
  }

  vector <string> filenames;
  
  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    cerr << "the_decimator: please specify a filename (-h for help)" 
	 << endl;
    return 0;
  }

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  if (verbose)
    cerr << "the_decimator: creating input timeseries container" << endl;
  Reference::To<dsp::TimeSeries> timeseries = new dsp::TimeSeries;

  if (verbose)
    cerr << "the_decimator: creating input/unpacker manager" << endl;
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;
  manager->set_output (timeseries);

  if (verbose)
    cerr << "the_decimator: creating BandpassMonitor" << endl;
  Reference::To<dsp::BandpassMonitor> monitor = new dsp::BandpassMonitor;

  if (verbose)
    cerr << "the_decimator: creating rescale transformation" << endl;
  Reference::To<dsp::Rescale> rescale = new dsp::Rescale;
  rescale->set_input (timeseries);
  rescale->set_output (timeseries);
  rescale->set_interval_seconds (update_interval);
  rescale->set_output_time_total (true);
  rescale->set_constant (constant_offset_scale);
  rescale->update.connect (monitor, &dsp::BandpassMonitor::output_state);

  if (verbose)
    cerr << "the_decimator: creating pscrunch transformation" << endl;
  Reference::To<dsp::PScrunch> pscrunch = new dsp::PScrunch;
  pscrunch->set_input (timeseries);
  pscrunch->set_output (timeseries);



  Reference::To<dsp::FScrunch> fscrunch = new dsp::FScrunch;
  Reference::To<dsp::TScrunch> tscrunch = new dsp::TScrunch;
  if ( fscrunch_factor )
  {

	  fscrunch->set_factor( fscrunch_factor );
	  fscrunch->set_input( timeseries );
	  fscrunch->set_output( timeseries );
  }

  if ( tscrunch_factor )
  {

	  tscrunch->set_factor( tscrunch_factor );
	  tscrunch->set_input( timeseries );
	  tscrunch->set_output( timeseries );
  }





  if (verbose)
    cerr << "the_decimator: creating output bitseries container" << endl;
  Reference::To<dsp::BitSeries> bitseries = new dsp::BitSeries;

  if (verbose)
    cerr << "the_decimator: creating sigproc digitizer" << endl;
  Reference::To<dsp::SigProcDigitizer> digitizer = new dsp::SigProcDigitizer;
  digitizer->set_nbit (nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);

#ifdef  SIGPROC_FILTERBANK_RINGBUFFER
  if (verbose)
    cerr << "the_decimator: creating 2nd sigproc digitizer for ringbuffer"
	 << endl;
  Reference::To<dsp::BitSeries> bitseries_rb = new dsp::BitSeries;    
  Reference::To<dsp::SigProcDigitizer> digitizer_rb = new dsp::SigProcDigitizer;
  digitizer_rb->set_nbit(8);
  digitizer_rb->set_input (timeseries);
  digitizer_rb->set_output (bitseries_rb);
#endif

#ifdef HAVE_PSRXML
  psrxml* psrxml_header =0;
  dataFile* rawDataFile = 0;
  dataBlockHeader* blockHeaders =0;
  int numberOfBlocksRecorded = 0;
  unsigned long long totalSamplesRecorded = 0;
  int blockHeaders_length = 2;

  if(write_psrxml){
	  // prepare the psrxml header 
	  psrxml_header = (psrxml*) malloc(sizeof(psrxml));
	  // blank the xml header
	  clearPsrXmlDoc(psrxml_header);

	  // create a dataFile to store the info for the raw data file.
	  rawDataFile = (dataFile*) malloc(sizeof(dataFile));
	  blockHeaders = (dataBlockHeader*)malloc(sizeof(dataBlockHeader)*blockHeaders_length);

  }

#endif

  bool written_header = false;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "the_decimator: opening file " << filenames[ifile] << endl;

    manager->open (filenames[ifile]);

    dsp::Observation* obs = manager->get_info();

    unsigned nchan = obs->get_nchan();
    uint64_t nsample = uint64_t( block_size * obs->get_rate() );

    if (verbose)
      cerr << "the_decimator: block_size=" << block_size << " sec "
	"(" << nsample << " samp)" << endl;

    manager->set_block_size( nsample );

    dsp::Unpacker* unpacker = manager->get_unpacker();

    if (unpacker->get_order_supported (order))
      unpacker->set_output_order (order);

    if (verbose)
    {
      cerr << "the_decimator: file " 
	   << filenames[ifile] << " opened" << endl;
      cerr << "Source = " << obs->get_source() << endl;
      cerr << "Frequency = " << obs->get_centre_frequency() << endl;
      cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
      cerr << "Sampling rate = " << obs->get_rate() << endl;
    }

    dsp::SigProcObservation sigproc;

#ifdef SIGPROC_FILTERBANK_RINGBUFFER

    dsp::ASCIIObservation ascii_hdr;
    bool written_rb_header = false;

    if (hdu_key)
    {
      multilog_t* mlog = multilog_open ("the_decimator",0);
      multilog_add (mlog, stderr);
      
      hdu = dada_hdu_create(mlog);
      hdu->header = (char*) malloc(4096);
      dada_hdu_set_key(hdu,hdu_key);
      if(dada_hdu_connect(hdu) < 0)
	return -1;
      if(dada_hdu_lock_write(hdu) < 0)
	return -1;
    }

#endif

    bool do_pscrunch = manager->get_info()->get_npol() > 1;
    uint64_t lost_samps = 0;
    while (!manager->get_input()->eod())
    {
      manager->operate ();

      rescale->operate ();

      if (do_pscrunch)
	      pscrunch->operate ();
      if ( fscrunch_factor )
	      fscrunch->operate();
      if ( tscrunch_factor )
	      tscrunch->operate();

#ifdef SIGPROC_FILTERBANK_RINGBUFFER
	
    if (hdu_key)
    {
      if (ipcio_space_left(hdu->data_block))
      {
	digitizer_rb->operate();
	if (!written_rb_header)
	{
	  ascii_hdr.copy(bitseries_rb);
	  char* buf = ipcbuf_get_next_write(hdu->header_block);
	  ascii_hdr.unload(buf);
	  ipcbuf_mark_filled(hdu->header_block,4096);
	  ipcbuf_unlock_write(hdu->header_block);
	  written_rb_header = true;
	}
	ipcio_write(hdu->data_block,(char*)bitseries_rb->get_rawptr(),bitseries_rb->get_nbytes());
      }
      else
      {
	lost_samps += timeseries->get_ndat();
      }
    }

#endif
    
    digitizer->operate ();

    if (!written_header)
    {
	    sigproc.copy( bitseries );
	    sigproc.unload( outfile );
	    written_header = true;
#ifdef HAVE_PSRXML
	    if(write_psrxml){
		    // set the basic parameters, undefined things will be set as we go along or at the end.
		    strcpy(psrxml_header->sourceName,timeseries->get_source().c_str());
		    strcpy(psrxml_header->sourceNameCentreBeam,""); // @todo
		    // cat reference is not yet defined
		    psrxml_header->mjdObs = timeseries->get_start_time().intday();
		    psrxml_header->timeToFirstSample = (unsigned long long)(timeseries->get_start_time().get_secs())*(unsigned long long)1000000000 + (unsigned long long)(timeseries->get_start_time().get_fracsec()*1e9);
		    strcpy(psrxml_header->utc,""); //@todo
		    strcpy(psrxml_header->lst,""); //@todo
		    strcpy(psrxml_header->localTime,""); //@todo
		    psrxml_header->nativeSampleRate = 1.0/(timeseries->get_rate());
		    psrxml_header->currentSampleInterval = psrxml_header->nativeSampleRate; // we don't do any resampling
		    // number of samples is not yet defined... 
		    psrxml_header->requestedObsTime = 0; //@todo
		    //actualObsTime is not yet defined.
		    psrxml_header->centreFreqCh1 = timeseries->get_centre_frequency(0);
		    psrxml_header->freqOffset =  timeseries->get_bandwidth() / timeseries->get_nchan();
		    if(timeseries->get_bandwidth() > 0){
			    // the band will be flipped by the sigproc digitizer, so update the header
			    psrxml_header->freqOffset = -psrxml_header->freqOffset;
			    psrxml_header->centreFreqCh1 = timeseries->get_centre_frequency(timeseries->get_nchan()-1);
		    }
		    psrxml_header->numberOfChannels =  timeseries->get_nchan();
		    psrxml_header->startCoordinate.ra = timeseries->get_coordinates().ra().getDegrees();
		    psrxml_header->startCoordinate.dec = timeseries->get_coordinates().dec().getDegrees();
		    strcpy(psrxml_header->startCoordinate.posn_epoch,"J2000");
		    // end coordinate not known...
		    psrxml_header->requestedCoordinate.ra=0; //@todo
		    psrxml_header->requestedCoordinate.dec=0; //@todo
		    strcpy(psrxml_header->requestedCoordinate.posn_epoch,"J2000");
		    psrxml_header->startParalacticAngle = 0; //@todo
		    // end paralactic angle not known
		    psrxml_header->isParalacticAngleTracking = 1; //@todo
		    psrxml_header->startAzEl.az=0; //@todo
		    psrxml_header->startAzEl.el=0; //@todo
		    strcpy(psrxml_header->observingProgramme,timeseries->get_identifier().c_str());
		    strcpy(psrxml_header->observerName,""); //@todo
		    strcpy(psrxml_header->observationType,Source2string(timeseries->get_type()).c_str());
		    strcpy(psrxml_header->observationConfiguration,"");//@todo
		    strcpy(psrxml_header->telescopeIdentifyingString,timeseries->get_telescope().c_str());
		    strcpy(psrxml_header->receiverIdentifyingString,timeseries->get_receiver().c_str());
		    strcpy(psrxml_header->backendIdentifyingString,timeseries->get_machine().c_str());

		    strcpy(psrxml_header->telescopeConfigString,"");//@todo
		    strcpy(psrxml_header->receiverIdentifyingString,"");//@todo
		    strcpy(psrxml_header->backendIdentifyingString,"");//@todo


		    strcpy(psrxml_header->receiver.name,timeseries->get_receiver().c_str());
		    psrxml_header->receiver.hasCircularFeeds = timeseries->get_basis() == Signal::Circular;
		    psrxml_header->receiver.feedRightHanded = 0; // @todo
		    psrxml_header->receiver.numberOfPolarisations = 2;
		    psrxml_header->receiver.feedSymetry = 0; // @todo
		    psrxml_header->receiver.calXYPhase = 0; //@todo

		    psrxml_header->receiverBeamNumber = 0;//@todo
		    psrxml_header->totalBeamsRecorded = 0;//@todo
		    // sky beam number not known

		    strcpy(psrxml_header->backend.name,timeseries->get_machine().c_str());
		    psrxml_header->backend.sigprocCode = 0; //@todo
		    psrxml_header->backend.upperSideband = timeseries->get_bandwidth() > 0; // assume that input data are 'raw'
		    psrxml_header->backend.reverseCrossPhase = 0; //@todo

		    strcpy(psrxml_header->recordedPol,"II");
		    strcpy(psrxml_header->observedPol,""); //@todo
		    psrxml_header->nRecordedPol = 1;

		    strcpy(psrxml_header->telescope.name,timeseries->get_telescope().c_str());
		    psrxml_header->telescope.longitude = 0;//@todo
		    psrxml_header->telescope.lattitude=0;//@todo
		    psrxml_header->telescope.zenithLimit=0;//@todo
		    psrxml_header->telescope.x=0;//@todo
		    psrxml_header->telescope.y=0;//@todo
		    psrxml_header->telescope.z=0;//@todo
		    psrxml_header->telescope.sigprocCode=0;//@todo
		    strcpy(psrxml_header->telescope.tempoCode,"");//@todo
		    strcpy(psrxml_header->telescope.pulsarhunterCode,"");//@todo

		    psrxml_header->comment = (char*)malloc(sizeof(char)*1024);
		    strcpy(psrxml_header->comment,"PsrXML written by The DECIMATOR");

		    rawDataFile->source = psrxml_header;
		    string filename = outfile_basename;
		    filename += ".fil";
		    strcpy(rawDataFile->filename,filename.c_str());
		    strcpy(rawDataFile->dataType,"TIMESERIES");
		    // uid etc not avaliable yet.
		    // checksum not avaliable yet.
		    rawDataFile->endian = INDEPENDANT;
		    rawDataFile->headerLength = ftell(outfile);
		    rawDataFile->blockLength = timeseries->get_ndat()*psrxml_header->numberOfChannels*nbits/8;
		    rawDataFile->blockHeaderLength=0;
		    rawDataFile->bitsPerSample = nbits;
		    rawDataFile->isChannelInterleaved = 1;
		    rawDataFile->firstSampleIsMostSignificantBit = 0;
		    rawDataFile->isSigned =0;
	    }
#endif


    }

    // output the result to stdout
    const uint64_t nbyte = bitseries->get_nbytes();
    unsigned char* data = bitseries->get_rawptr();

    fwrite (data,nbyte,1,outfile);

#ifdef HAVE_PSRXML

    totalSamplesRecorded += timeseries->get_ndat();
    if(write_psrxml){
	    if(numberOfBlocksRecorded >= blockHeaders_length){
		    blockHeaders_length*=2;
		    blockHeaders = (dataBlockHeader*) realloc(blockHeaders,sizeof(dataBlockHeader)*blockHeaders_length);
	    }
#ifdef HAVE_OPENSSL_SHA_H

	    get_SHA_hash(data,nbyte,blockHeaders[numberOfBlocksRecorded].sha1_hash);    
	    blockHeaders[numberOfBlocksRecorded].has_sha1_hash=1;
    }
#endif
    numberOfBlocksRecorded++;

#endif

    }

#ifdef SIGPROC_FILTERBANK_RINGBUFFER
    if (hdu_key)
    {
      ipcio_close(hdu->data_block);
      fprintf(stderr,"Downwind processes lost %lld samps due to buffer overrun\n",lost_samps);
    }

#endif

    fclose(outfile);

#ifdef HAVE_PSRXML
    if(write_psrxml){
	    rawDataFile->blockHeaders_length = numberOfBlocksRecorded;
	    rawDataFile->blockHeaders = blockHeaders;

	    psrxml_header->numberOfSamples = totalSamplesRecorded;
	    psrxml_header->actualObsTime = totalSamplesRecorded*psrxml_header->currentSampleInterval;

	    psrxml_header->files_length = 1;

	    psrxml_header->files = (dataFile**) malloc(sizeof(dataFile*)*psrxml_header->files_length); 
	    psrxml_header->files[0] = rawDataFile;

	    char header_filename[80];
	    sprintf(header_filename,"%s.psrxml",outfile_basename);
	    writePsrXml(psrxml_header,header_filename);
	    freePsrXml(psrxml_header);
    }
#endif

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

  }
  catch (Error& error)
  {
    cerr << error << endl;
  }

  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}




#ifdef HAVE_OPENSSL_SHA_H
char get_SHA_hash(unsigned char* buffer,int size, char* hashStr) {
        unsigned char hash[SHA_DIGEST_LENGTH];
        char *ptr;
        int i;

        SHA1(buffer, size, hash);
        ptr = hashStr;
        for (i = 0; i < SHA_DIGEST_LENGTH; i++) {
                sprintf(ptr, "%02x", hash[i]);
                ptr+=2;

        }
        return 1;
}
#endif

