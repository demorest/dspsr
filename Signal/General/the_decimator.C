/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcObservation.h"
#include "dsp/SigProcDigitizer.h"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"

#include "dsp/Rescale.h"
#include "dsp/PScrunch.h"

#include "dirutil.h"
#include "Error.h"

#include <iostream>
#include <unistd.h>
#include <string.h>

#define SIGPROC_FILTERBANK_RINGBUFFER
#ifdef SIGPROC_FILTERBANK_RINGBUFFER
#include <dada_hdu.h>
#include <ipcio.h>
#include "dsp/ASCIIObservation.h"
#endif

using namespace std;

static char* args = "b:B:o:prhk:vV";

void usage ()
{
  cout << "sigproc_filterbank - convert dspsr input to sigproc filterbank \n"
    "Usage: sigproc_filterbank file1 [file2 ...] \n"
    "Options:\n"
    "\n"
    "  -b bits   number of bits per sample output to file \n" 
    "  -B samps  number of samples per block \n"
    "  -o file   file stamp for filterbank file  \n" 
    "  -r        report total Operation times \n"
    "  -p        revert to PFT order \n"
#ifdef SIGPROC_FILTERBANK_RINGBUFFER
    "  -k key    shared memory key to output DADA ring buffer \n"
#endif
       << endl;
}

int main (int argc, char** argv) try 
{
  bool verbose = false;
  int nbits = 8;

#ifdef SIGPROC_FILTERBANK_RINGBUFFER
  dada_hdu_t* hdu = 0;
  key_t hdu_key = 0;
#endif

  // a kilo-sample at a time
  uint64 block_size = 1024;

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
      block_size = atoi (optarg);
      break;

    case 'o':
      outfile_basename = optarg;
      break;

    case 'p':
      order = dsp::TimeSeries::OrderPFT;
      break;

    case 'r':
      dsp::Operation::record_time = true;
      break;

    case 'k':
      if (sscanf (optarg, "%x", &hdu_key) != 1)
      {
	cerr << "sigproc_filterbank: could not scan key from "
	  "'" << hdu_key << "'" << endl;
	return -1;
      }
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
    cerr << "sigproc_filterbank: please specify a filename (-h for help)" 
	 << endl;
    return 0;
  }

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  if (verbose)
    cerr << "sigproc_filterbank: creating input timeseries container" << endl;
  Reference::To<dsp::TimeSeries> timeseries = new dsp::TimeSeries;

  if (verbose)
    cerr << "sigproc_filterbank: creating input/unpacker manager" << endl;
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;
  manager->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: creating rescale transformation" << endl;
  Reference::To<dsp::Rescale> rescale = new dsp::Rescale;
  rescale->set_input (timeseries);
  rescale->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: attaching BandpassMonitor" << endl;
  Reference::To<dsp::BandpassMonitor> monitor = new dsp::BandpassMonitor;

  rescale->update.connect( monitor, &dsp::BandpassMonitor::output_state);

  if (verbose)
    cerr << "sigproc_filterbank: creating pscrunch transformation" << endl;
  Reference::To<dsp::PScrunch> pscrunch = new dsp::PScrunch;
  pscrunch->set_input (timeseries);
  pscrunch->set_output (timeseries);

  if (verbose)
    cerr << "sigproc_filterbank: creating output bitseries container" << endl;
  Reference::To<dsp::BitSeries> bitseries = new dsp::BitSeries;

  if (verbose)
    cerr << "sigproc_filterbank: creating sigproc digitizer" << endl;
  Reference::To<dsp::SigProcDigitizer> digitizer = new dsp::SigProcDigitizer;
  digitizer->set_nbit (nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);

#ifdef  SIGPROC_FILTERBANK_RINGBUFFER
  if (verbose)
    cerr << "sigproc_filterbank: creating 2nd sigproc digitizer for ringbuffer"
	 << endl;
  Reference::To<dsp::BitSeries> bitseries_rb = new dsp::BitSeries;    
  Reference::To<dsp::SigProcDigitizer> digitizer_rb = new dsp::SigProcDigitizer;
  digitizer_rb->set_nbit(8);
  digitizer_rb->set_input (timeseries);
  digitizer_rb->set_output (bitseries_rb);
#endif

  bool written_header = false;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    if (verbose)
      cerr << "sigproc_filterbank: opening file " << filenames[ifile] << endl;

    manager->open (filenames[ifile]);

    unsigned nchan = manager->get_info()->get_nchan();

    manager->get_input()->set_block_size( block_size );

    dsp::Unpacker* unpacker = manager->get_unpacker();

    if (unpacker->get_order_supported (order))
      unpacker->set_output_order (order);

    if (verbose)
    {
      dsp::Observation* obs = manager->get_info();

      cerr << "sigproc_filterbank: file " 
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
      multilog_t* mlog = multilog_open ("sigproc_filterbank",0);
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
    uint64 lost_samps = 0;
    while (!manager->get_input()->eod())
    {
      manager->operate ();

      rescale->operate ();

      if (do_pscrunch)
	pscrunch->operate ();

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
    }

    // output the result to stdout
    const uint64 nbyte = bitseries->get_nbytes();
    unsigned char* data = bitseries->get_rawptr();

    fwrite (data,nbyte,1,outfile);

    }

#ifdef SIGPROC_FILTERBANK_RINGBUFFER
    if (hdu_key)
    {
      ipcio_close(hdu->data_block);
      fprintf(stderr,"Downwind processes lost %lld samps due to buffer overrun\n",lost_samps);
    }

#endif

    fclose(outfile);

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




