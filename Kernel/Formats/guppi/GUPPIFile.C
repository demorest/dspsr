/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPIFile.h"

#include "Error.h"

#include <sys/stat.h>
#include <fcntl.h>

#include "ascii_header.h"

using namespace std;

dsp::GUPPIFile::GUPPIFile (const char* filename)
  : BlockFile ("GUPPI")
{
  hdr = NULL;
}

dsp::GUPPIFile::~GUPPIFile ( )
{
  if (hdr!=NULL) free(hdr);
}

// Read header starting from current spot in file.  Puts header info
// into hdr buf, and returns total number of keys found.  If no 
// "END" is found, frees hdr and return 0.
int get_header(int fd, char **hdr) {
  const size_t cs = 80;
  char card[cs];  // Note these do not have terminating nulls!
  char end[cs];   // No null here either!
  memset(end, ' ', cs);
  strncpy(end, "END", 3);
  const int max_cards = 2304;
  int count=0;
  bool got_end=false;
  while (count<max_cards && !got_end) {

    // Read next card
    unsigned rv = read(fd, card, cs);
    if (rv!=cs)
      throw Error (FailedSys, "dsp::GUPPIFile::get_header", 
          "read() failed");

    // Check for END
    if (strncmp(card,end,cs)==0) 
      got_end = true;

    count++;

    // Copy into hdr buf
    *hdr = (char *)realloc(*hdr, count*cs + 1);
    strncpy(&(*hdr)[(count-1)*cs], card, cs);
    (*hdr)[count*cs] = '\0';
  }

  if (got_end==false) {
    free(*hdr);
    *hdr = NULL;
    return 0;
  } 

  // Strip out the '=' so we can use dspsr's parsing routine later.
  for (int i=0; i<count; i++) {
    if ((*hdr)[i*cs+8]=='=') (*hdr)[i*cs+8]=' ';
    // TODO make this an error if the '=' is not present?
  }

  // Strip out single quotes for parsing.. may cause problems
  // for strings with spaces.
  for (int i=0; i<count*cs; i++) {
    if ((*hdr)[i]=='\'') (*hdr)[i]=' ';
  }

  return count;

}

bool dsp::GUPPIFile::is_valid (const char* filename) const
{

  int fd = ::open(filename, O_RDONLY);
  if (fd < 0) 
    throw Error (FailedSys, "dsp::GUPPIFile::is_valid",
        "fopen(%s) failed", filename);

  // Try reading header
  char *hdr_tmp=NULL;
  int nkeys = get_header(fd, &hdr_tmp);
  ::close(fd);
  if (nkeys<=0) {
    if (verbose)
      cerr << "dsp::GUPPIFile couldn't parse header." << endl;
    return false;
  }

  // If header read ok, check for some required keywords
  // that don't appear in psrfits.
  int rv, itmp;

  rv = ascii_header_get(hdr_tmp, "BLOCSIZE", "%d", &itmp);
  if (rv<0) {
    if (verbose)
      cerr << "dsp::GUPPIFile coulnd't find BLOCSIZE keyword." << endl;
    return false;
  }

  rv = ascii_header_get(hdr_tmp, "PKTIDX", "%d", &itmp);
  if (rv<0) {
    if (verbose)
      cerr << "dsp::GUPPIFile coulnd't find PKTIDX keyword." << endl;
    return false;
  }

  // Everything passed, it's probably a GUPPI file.
  return true;
}

// Use for required header params
#define header_get_check(param, fmt, var) { \
  rv = ascii_header_get(hdr, param, fmt, var); \
  if (rv<=0) \
    throw Error (InvalidState, "dsp::GUPPIFile::open_file", \
        "Couldn't find %s keyword in header.", param); \
}

void dsp::GUPPIFile::open_file (const char* filename)
{

  // Open file
  fd = ::open(filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::GUPPIFile::open_file",
        "open(%s) failed", filename);

  // Read in header
  hdr_keys = get_header(fd, &hdr);
  if (hdr_keys<=0) 
    throw Error (InvalidState, "dsp::GUPPIFile::open_file",
        "Error parsing header.");
 
  // Read header params
  int rv, itmp;
  float ftmp;
  char ctmp[80];

  header_get_check("NBIT", "%d", &itmp);
  info.set_nbit(itmp);

  header_get_check("OBSBW", "%f", &ftmp);
  info.set_bandwidth(ftmp);

  header_get_check("OBSFREQ", "%f", &ftmp);
  info.set_centre_frequency(ftmp);
 
  header_get_check("OBSNCHAN", "%d", &itmp);
  info.set_nchan(itmp);
 
  // Assume we have baseband data?
  info.set_npol(2);
  info.set_state(Signal::Analytic);

  header_get_check("TBIN", "%f", &ftmp);
  info.set_rate(1.0/ftmp);

  int imjd, smjd;
  double t_offset;
  header_get_check("STT_IMJD", "%d", &imjd);
  header_get_check("STT_SMJD", "%d", &smjd);
  header_get_check("STT_OFFS", "%lf", &t_offset);
  MJD epoch (imjd, (double)smjd/86400.0 + t_offset);
  info.set_start_time( epoch );

  header_get_check("TELESCOP", "%s", ctmp);
  info.set_telescope(ctmp);

  header_get_check("SRC_NAME", "%s", ctmp);
  info.set_source(ctmp);

  // Header, data sizes per block.
  // TODO: Assume header size doesn't change?
  //       What about overlap?
  header_bytes = 0;
  block_header_bytes = 80*hdr_keys;
  block_tailer_bytes = 0;
  header_get_check("BLOCSIZE", "%d", &itmp);
  block_bytes = itmp + block_header_bytes;

  set_total_samples();

  header_get_check("FD_POLN", "%s", ctmp);
  info.set_mode(ctmp);
  header_get_check("BACKEND", "%s", ctmp);
  info.set_machine(ctmp);

  // TODO: could set recvr, etc..
  
}

void dsp::GUPPIFile::skip_extra ()
{
  // We should be at a new header now
  int nkeys = get_header(fd, &hdr);
  if (nkeys!=hdr_keys)
    throw Error (InvalidState, "dsp::GUPPIFile::skip_extra", 
        "Number of header keys changed, can't deal with this yet.");
}

