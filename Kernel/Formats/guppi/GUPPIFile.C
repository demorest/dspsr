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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
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
    if (rv<0) // Error
      throw Error (FailedSys, "dsp::GUPPIFile::get_header", 
          "read() failed");
    else if (rv!=cs) // Probably EOF
      break;

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
  uint64_t ltmp;
  double ftmp;
  char ctmp[80], ctmp2[80];

  header_get_check("NBIT", "%d", &itmp);
  info.set_nbit(itmp);

  header_get_check("OBSBW", "%lf", &ftmp);
  info.set_bandwidth(ftmp);

  header_get_check("OBSFREQ", "%lf", &ftmp);
  info.set_centre_frequency(ftmp);
 
  header_get_check("OBSNCHAN", "%d", &itmp);
  info.set_nchan(itmp);
 
  // Assume we have baseband data?
  info.set_npol(2);
  info.set_state(Signal::Analytic);

  header_get_check("TBIN", "%lf", &ftmp);
  info.set_rate(1.0/ftmp);

  int imjd, smjd;
  double t_offset;
  header_get_check("STT_IMJD", "%d", &imjd);
  header_get_check("STT_SMJD", "%d", &smjd);
  header_get_check("STT_OFFS", "%lf", &t_offset);
  header_get_check("PKTIDX", "%lld", &ltmp);
  header_get_check("PKTSIZE", "%d", &itmp);
  t_offset += ltmp * itmp * 8.0 / info.get_rate() / 
      (info.get_nbit() * info.get_nchan() * info.get_npol() * 2.0);
  //cerr << "t_offset=" << t_offset << "s" << endl;
  MJD epoch (imjd, (double)smjd/86400.0 + t_offset/86400.0);
  info.set_start_time(epoch);

  header_get_check("TELESCOP", "%s", ctmp);
  info.set_telescope(ctmp);

  header_get_check("SRC_NAME", "%s", ctmp);
  info.set_source(ctmp);

  // Header, data sizes per block.
  // TODO: Assume header size doesn't change?
  header_bytes = 0;
  block_header_bytes = 80*hdr_keys;
  header_get_check("OVERLAP", "%d", &itmp);
  block_tailer_bytes = itmp*info.get_nchan()*4; // Assumes 8-bit, 2-pol
  header_get_check("BLOCSIZE", "%d", &itmp);
  block_bytes = itmp + block_header_bytes;

  set_total_samples();

  header_get_check("BACKEND", "%s", ctmp);
  info.set_machine(ctmp);

  // Maybe the following aren't strictly required ...

  // Poln type
  header_get_check("FD_POLN", "%s", ctmp);
  if (strncasecmp(ctmp, "CIR", 3)==0) 
    info.set_basis(Signal::Circular);
  else
    info.set_basis(Signal::Linear);

  // Coordinates
  sky_coord coords;
  header_get_check("RA_STR", "%s", ctmp);
  header_get_check("DEC_STR", "%s", ctmp2);
  coords.setHMSDMS(ctmp, ctmp2);
  info.set_coordinates(coords);

  // Receiver
  header_get_check("FRONTEND", "%s", ctmp);
  info.set_receiver(ctmp);
  // How to set feed hand, symm angle, etc?
  // Note: GBT recvrs have fd_hand=-1, PF has fd_sang=+45deg, 
  //       otherwise fd_sang=-45deg.
  
}

void dsp::GUPPIFile::skip_extra ()
{
  if (verbose) 
    cerr << "dsp::GUPPIFile::skip_extra()" << endl;
  // We should be at a new header now
  int nkeys = get_header(fd, &hdr);
  if (nkeys>0 && nkeys!=hdr_keys)
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
    uint64_t start_pos = lseek(fd, 0, SEEK_CUR);
    bool eof = false;
    for (unsigned ichan=0; ichan<nchan; ichan++) {

      // Go to this channel's spot
      uint64_t offset = start_pos + (uint64_t)ichan*full_block_bytes_per_chan;
      int64_t rv = lseek(fd, offset, SEEK_SET);
      if (rv<0) 
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", 
            "lseek1(%d) ichan=%d start_pos=%lld full_bytes_per_chan=%lld off=%lld", 
            fd, ichan, start_pos, full_block_bytes_per_chan, offset);

      // Read the data
      size_t bytes_read = read(fd, tmpbuf + ichan*to_load_per_chan + 
          bytes_read_per_chan, to_read_per_chan);

      if (bytes_read < 0)
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", "read");
                //"read (got=%d expected=%d)",
                //bytes_read, to_read_per_chan);

      if (bytes_read < to_read_per_chan) {
          eof = true;
          break;
      }

    }

    if (eof) break;

    // Go back to right spot in file
    lseek(fd, start_pos + to_read_per_chan, SEEK_SET);

    // Increment read counter
    bytes_read_per_chan += to_read_per_chan;
    to_load -= to_read_per_chan*nchan;
    current_block_byte += to_read_per_chan*nchan;

    // Go to next block
    if (current_block_byte == block_data_bytes) {

      int64_t rv = lseek(fd, 
          full_block_bytes_per_chan*(nchan-1) + block_tailer_bytes_per_chan, 
          SEEK_CUR);
      if (rv<0)
        throw Error (FailedSys, "dsp::GUPPIFile::load_bytes", "lseek2(%d)", fd);

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
