#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

// CPSR header and unpacking routines
#define cpsr 1
#include "pspm_search_header.h"
#include "pspmDbase.h"

#include "CPSRFile.h"
#include "pspm++.h"
#include "genutil.h"

bool dsp::CPSRFile::is_valid (const char* filename) const
{
  int fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    return false;

  PSPM_SEARCH_HEADER* header = pspm_read (fd);
  ::close (fd);

  if (!header)
    return false;

  return true;
}

//! Construct and open file
dsp::CPSRFile::CPSRFile (const char* filename)
{ 
  tapenum = filenum = -1;
  if (filename)
    open (filename);
}

// #define _DEBUG 1

//
// NEW! 1 Aug 01
// It has been determined that PSPM headers are coming off DLT tape
// corrupted.   Luckily, a copy of most headers can be found on disk
// on orion.  An ascii database is created from these files, and
// served by the pspmDbase::server and pspmDbase::entry clases

static pspmDbase::server cpsr_hdr;

/* ***********************************************************************
   load

   Construct a Bit_Stream object from a CPSR (PSPM) style file.

   If tape_num or file_num are not NULL,
   the tape number and file number (on tape) will be returned via the
   arguments to which they point.

   *********************************************************************** */

// NOTE!!!!!
//
// If you change the way this stuff works, please try to make sure
// the multi-file constructor (below) will still work.

void dsp::CPSRFile::open (const char* filename)
{
  if (verbose)
    cerr << "CPSRFile::open " << filename << endl;

  if ( sizeof(PSPM_SEARCH_HEADER) != PSPM_HEADER_SIZE ) {
    fprintf (stderr, "CPSRFile:: PSPM header size is invalid.\n");
    fprintf (stderr, "CPSRFile:: PSPM header size %d.\n", PSPM_HEADER_SIZE);
    fprintf (stderr, "CPSRFile:: for this architecture: %d.\n",
	     sizeof(PSPM_SEARCH_HEADER));

    throw_str ("CPSRFile::open - Architecture Error!");
  }

  header_bytes = sizeof(PSPM_SEARCH_HEADER);

  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw_str ("CPSRFile::open - failed open(%s) %s\n",
	       filename, strerror(errno));

  PSPM_SEARCH_HEADER* header = pspm_read (fd);
  if (!header) {
    ::close (fd);
    throw_str ("CPSRFile::open - failed pspm_read(%s) %s\n", 
	     filename, strerror(errno));
  }

#if _DEBUG

  fprintf (stderr, "File size:       %ld\n", header->file_size);
  fprintf (stderr, "Large File Size: "I64"\n", header->ll_file_size);
  fprintf (stderr, "Large Offset:    "I64"\n", header->ll_file_offset);
  fprintf (stderr, "MJD in header    %40.38Lf\n", header->mjd_start);

  fprintf (stderr, "tick offset:     %30.28lf\n\n", header->tick_offset);
  fprintf (stderr, "tape_num:        %d\n", header->tape_num);
  fprintf (stderr, "tape_file_number:%d\n", header->tape_file_number);
  fprintf (stderr, "scan_num:        %d\n", header->scan_num);
  fprintf (stderr, "scan_file_number %d\n", header->scan_file_number);
  fprintf (stderr, "file_size        %d\n\n", header->file_size);

  fprintf (stderr, "LMST in header   %lf\n", header->pasmon_lmst);

  fprintf (stderr, "Pulsar:          %s\n",  header->psr_name);
  fprintf (stderr, "Date:            %s\n",  header->date);
  fprintf (stderr, "Start Time:      %s\n",  header->start_time);

  fprintf (stderr, "pasmon_daynumber:%d\n",  header->pasmon_daynumber);
  fprintf (stderr, "pasmon_ast:      %d\n",  header->pasmon_ast);
 
  fprintf (stderr, "Centre Freq:     %lf\n", header->rf_freq);
  fprintf (stderr, "Sampling Period: %lf\n", header->samp_rate);
  fprintf (stderr, "Bandwidth:       %lf\n", header->bw);
  fprintf (stderr, "SIDEBAND:        %d:",  header->SIDEBAND);
  switch (header->SIDEBAND) 
    {
    case UNKNOWN_SIDEBAND:
      fprintf (stderr, "Unknown (assume USB)\n");
      break;
    case SSB_LOWER:
      fprintf (stderr, "SSB Lower\n");
      break;
    case SSB_UPPER:
      fprintf (stderr, "SSB Upper\n");
      break;
    case DSB_SKYFREQ:
      fprintf (stderr, "DSB Sky frequency order\n");
      break;
    case DSB_REVERSED:
      fprintf (stderr, "DSB Sky reversed frequency order\n");
      break;
    default:
      fprintf (stderr, "Internal error\n");
      break;
    }
  
  fprintf (stderr, "Telescope:       %d\n",  header->observatory);
  fprintf (stderr, "Channels:        %ld\n", header->num_chans);
  fprintf (stderr, "Bit Mode:        %ld\n", header->bit_mode);
  
  fprintf (stderr, "CPSRFile::open - header size %d\n", header_bytes);

  fflush (stderr);
#endif

  pspmDbase::entry hdr;
  try { hdr = pspmDbase::Entry (header); }
  catch (...) {
    ::close (fd);
    throw_str ("CPSRFile::open no match found in pspmDbase for %s",
	       PSPMidentifier (header).c_str());
  }

  tapenum = hdr.tape;
  filenum = hdr.file;

  char modestr[30];
  sprintf (modestr, "%d", hdr.nbit);
  info.set_mode (modestr);

  info.set_start_time (hdr.start);

  /* redwards --- set "position" from the header.. should replace cal_ra/dec*/
  double ra_deg, dec_deg;
  ra_deg = 15.0 * floor(header->user_ra/10000.0)
    + 15.0/60.0 * floor(fmod(header->user_ra,10000.0)/100.0)
    + 15.0/3600.0 * fmod(header->user_ra,100.0);
  if (header->user_dec == 0.0)
    dec_deg = 0.0;
  else
    dec_deg = header->user_dec/fabs(header->user_dec)* // sign
      ( floor(fabs(header->user_dec)/10000.0)         // magnitude
	+1.0/60.0*floor(fmod(fabs(header->user_dec),10000.0)/100.0)
	+1.0/3600.0*fmod(fabs(header->user_dec),100.0));
  
  info.set_coordinates (sky_coord(ra_deg, dec_deg));

  info.set_source (hdr.name);

  /* IMPORTANT: tsamp is the sampling period in microseconds */
  info.set_rate (1e6/hdr.tsamp);
  info.set_bandwidth (hdr.bandwidth);

  // IMPORTANT: both telescope and centre_freq should be set before calling
  // default_basis
  info.set_telescope (hdr.ttelid);
  info.set_centre_frequency (hdr.frequency);
  info.set_default_basis();

  // CPSR samples the analytic representation of the voltages
  info.set_state (Signal::Analytic);
  // number of channels = number of polarizations * ((I and Q)=2);
  info.set_npol (hdr.ndigchan / 2);
  info.set_nbit (hdr.nbit);

  total_size = hdr.ndat;

  if (total_size < 1) {
    ::close (fd);
    throw_str ("CPSRFile::open - Total data: %d\n", total_size);
  }

  info.set_machine ("CPSR");

  // make an identifier name
  info.set_identifier ("o" + info.get_default_id());  // o for orion

  // set the file pointers
  reset();

  if (verbose)
    cerr << "CPSRFile::open exit" << endl;
}

