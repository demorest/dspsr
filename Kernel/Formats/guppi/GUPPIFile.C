/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPIFile.h"

#include "Error.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ascii_header.h"

using namespace std;

dsp::GUPPIFile::GUPPIFile (const char* filename)
  : BlockFile ("GUPPI")
{
  hdr = NULL;
  tmpbuf = NULL;
}

dsp::GUPPIFile::~GUPPIFile ( )
{
  if (hdr!=NULL) free(hdr);
  if (tmpbuf!=NULL) free(tmpbuf);
}

// Read header starting from current spot in file.  Puts header info
// into hdr buf, and returns total number of keys found.  If no 
// "END" is found, frees hdr and return 0.
int get_header(int fd, char **hdr) {
  const size_t cs = 80;
  char card[cs+1];
  char end[cs+1];
  card[cs] = '\0';
  end[cs] = '\0';
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

    // Print line for debug
    // cerr << card << endl;

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

  // Strip out single quotes for parsing, replace spaces with _
  // TODO this is kind of hacky.. figure out how to do it better/easier.
  bool in_quoted_string = false;
  for (int i=0; i<count*cs; i++) {
    if ((*hdr)[i]=='\'')  { 
        (*hdr)[i]=' '; 
        in_quoted_string = !in_quoted_string; 
    }
    //if (in_quoted_string && (*hdr)[i]==' ') (*hdr)[i]='_';
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
  info.set_start_time(epoch);

  header_get_check("TELESCOP", "%s", ctmp);
  info.set_telescope(ctmp);

  header_get_check("SRC_NAME", "%s", ctmp);
  info.set_source(ctmp);

  // Header, data sizes per block.
  // TODO: Assume header size doesn't change?
  //       What about overlap?
  header_bytes = 0;
  block_header_bytes = 80*hdr_keys;
  header_get_check("OVERLAP", "%d", &itmp);
  block_tailer_bytes = itmp*info.get_nchan()*4; // Assumes 8-bit, 2-pol
  header_get_check("BLOCSIZE", "%d", &itmp);
  block_bytes = itmp + block_header_bytes;

  set_total_samples();

  header_get_check("FD_POLN", "%s", ctmp);
  info.set_mode(ctmp);
  header_get_check("BACKEND", "%s", ctmp);
  info.set_machine(ctmp);

  info.set_dc_centred(false);
  info.set_swap(false);
  info.set_dual_sideband(false);

  // TODO: could set recvr, etc..
  
}

void dsp::GUPPIFile::skip_extra ()
{
  if (verbose) 
    cerr << "dsp::GUPPIFile::skip_extra()" << endl;
  // We should be at a new header now
  int nkeys = get_header(fd, &hdr);
  if (nkeys!=hdr_keys)
    throw Error (InvalidState, "dsp::GUPPIFile::skip_extra", 
        "Number of header keys changed (old=%d, new=%d), can't deal with this yet.",
        hdr_keys, nkeys);
}

//! Load bytes from file.
// This "untransposes" the GUPPI block structure... necessary??
// Based on original from BlockFile
int64_t dsp::GUPPIFile::load_bytes (unsigned char *buffer, uint64_t nbytes)
{
  if (verbose) 
    cerr << "dsp::GUPPIFile::load_bytes() nbytes=" << nbytes << endl;

  const uint64_t block_data_bytes = get_block_data_bytes();
  uint64_t to_load = nbytes;

  // Need to know nchan to do the transpose
  // Only considers 8-bit, 2-pol data now..
  const unsigned nchan = info.get_nchan();

  const uint64_t block_data_bytes_per_chan = block_data_bytes / nchan;
  uint64_t to_load_per_chan = to_load / nchan;
  uint64_t bytes_read_per_chan = 0;

  const uint64_t block_tailer_bytes_per_chan = block_tailer_bytes / nchan;
  const uint64_t full_block_bytes = block_data_bytes + block_tailer_bytes;
  const uint64_t full_block_bytes_per_chan = full_block_bytes / nchan;

  tmpbuf = (unsigned char *)realloc(tmpbuf, to_load);

  while (to_load) {

    // Here current_block_byte is not the location in the file, 
    // but the number of bytes that have been read from the
    // current block.
    uint64_t to_read = block_data_bytes - current_block_byte;
    if (to_read > to_load)
      to_read = to_load;

    // Hopefully to_read is always a multiple of nchan
    uint64_t to_read_per_chan = to_read / nchan;

    // Jump around in the file and get a bit of data from each channel
    // Assume we start at the current spot for chan 0.
    off_t start_pos = lseek(fd, 0, SEEK_CUR);
    for (unsigned ichan=0; ichan<nchan; ichan++) {

      // Go to this channel's spot
      int rv = lseek(fd, start_pos + ichan*full_block_bytes_per_chan, SEEK_SET);
      if (rv<0) 
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", "lseek(%d)", fd);

      // Read the data
      size_t bytes_read = read(fd, tmpbuf + ichan*to_load_per_chan + 
          bytes_read_per_chan, to_read_per_chan);

      if (bytes_read < to_read_per_chan)
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", "read");

    }

    // Go back to right spot in file
    lseek(fd, start_pos + to_read_per_chan, SEEK_SET);

    // Increment read counter
    bytes_read_per_chan += to_read_per_chan;
    to_load -= to_read_per_chan*nchan;
    current_block_byte += to_read_per_chan*nchan;

    // Go to next block
    if (current_block_byte == block_data_bytes) {

      int rv = lseek(fd, 
          full_block_bytes_per_chan*(nchan-1) + block_tailer_bytes_per_chan, 
          SEEK_CUR);
      if (rv<0)
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", "lseek(%d)", fd);

      skip_extra();
      current_block_byte = 0;

    }

  }

#if 1 
  // Transpose out of PTF into FPT(?) order
  const int npol = 2;
  const size_t bps = 2;
  const int nsamp = to_load_per_chan/bps/npol;
  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned isamp=0; isamp<nsamp; isamp++) {
      for (unsigned ipol=0; ipol<npol; ipol++) {
        memcpy(buffer + isamp*nchan*npol*bps + ipol*nchan*bps + ichan*bps,
            tmpbuf + ichan*nsamp*npol*bps + isamp*npol*bps + ipol*bps,
            bps);
      }
    }
  }
#endif

  return nbytes - to_load;

}
