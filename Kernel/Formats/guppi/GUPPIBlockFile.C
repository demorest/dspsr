/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPIBlockFile.h"

#include "Error.h"

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "fitshead.h"
#include "fitshead_utils.h"

using namespace std;

dsp::GUPPIBlockFile::GUPPIBlockFile (const char* name)
  : File (name)
{
  hdr = NULL;
  dat = NULL;
  time_ordered = true;
  current_block_byte = 0;
  overlap = 0;
  blocsize = 0;
}

dsp::GUPPIBlockFile::~GUPPIBlockFile ( )
{
}

// Use for required header params
#define header_get_check(param, var) {\
  rv = hget(hdr, param, var); \
  if (rv==0) \
    throw Error (InvalidState, "dsp::GUPPIBlockFile::parse_header", \
        "Couldn't find %s keyword in header.", param); \
}

void dsp::GUPPIBlockFile::parse_header()
{

  // Read header params
  int rv, itmp;
  long long ltmp;
  double ftmp;
  char ctmp[80], ctmp2[80];

  header_get_check("NBIT", &itmp);
  info.set_nbit(itmp);

  header_get_check("OBSBW", &ftmp);
  info.set_bandwidth(ftmp);

  header_get_check("OBSFREQ", &ftmp);
  info.set_centre_frequency(ftmp);
 
  header_get_check("OBSNCHAN", &itmp);
  info.set_nchan(itmp);
 
  // Assume we have baseband data?
  info.set_npol(2);
  info.set_state(Signal::Analytic);

  header_get_check("TBIN", &ftmp);
  info.set_rate(1.0/ftmp);

  int imjd, smjd;
  double t_offset;
  header_get_check("STT_IMJD", &imjd);
  header_get_check("STT_SMJD", &smjd);
  header_get_check("STT_OFFS", &t_offset);
  header_get_check("PKTIDX",   &ltmp);
  header_get_check("PKTSIZE",  &itmp);
  t_offset += ltmp * itmp * 8.0 / info.get_rate() / 
      (info.get_nbit() * info.get_nchan() * info.get_npol() * 2.0);
  //cerr << "t_offset=" << t_offset << "s" << endl;
  MJD epoch (imjd, (double)smjd/86400.0 + t_offset/86400.0);
  info.set_start_time(epoch);

  header_get_check("TELESCOP", ctmp);
  info.set_telescope(ctmp);

  header_get_check("SRC_NAME", ctmp);
  info.set_source(ctmp);

  // Data block size params
  header_get_check("OVERLAP", &itmp);
  overlap = itmp;
  header_get_check("BLOCSIZE", &itmp);
  blocsize = itmp;

  header_get_check("BACKEND", ctmp);
  info.set_machine(ctmp);

  // Maybe the following aren't strictly required ...

  // Poln type
  header_get_check("FD_POLN", ctmp);
  if (strncasecmp(ctmp, "CIR", 3)==0) 
    info.set_basis(Signal::Circular);
  else
    info.set_basis(Signal::Linear);

  // Coordinates
  sky_coord coords;
  header_get_check("RA_STR", ctmp);
  header_get_check("DEC_STR", ctmp2);
  coords.setHMSDMS(ctmp, ctmp2);
  info.set_coordinates(coords);

  // Receiver
  header_get_check("FRONTEND", ctmp);
  info.set_receiver(ctmp);
  // How to set feed hand, symm angle, etc?
  // Note: GBT recvrs have fd_hand=-1, PF has fd_sang=+45deg, 
  //       otherwise fd_sang=-45deg.
  
}

//! Send data bytes to unpacker
// This "untransposes" the GUPPI block structure if necessary.
// TODO: figure out how to not need untranspose?
int64_t dsp::GUPPIBlockFile::load_bytes (unsigned char *buffer, uint64_t nbytes)
{
  if (verbose) 
    cerr << "dsp::GUPPIBlockFile::load_bytes() nbytes=" << nbytes << endl;

  const unsigned nchan = info.get_nchan();
  const unsigned npol = info.get_npol();
  const unsigned nbit = info.get_nbit();
  const unsigned bytes_per_samp = (2 * nchan * npol * nbit) / 8;
  const uint64_t overlap_bytes = overlap * bytes_per_samp;
  uint64_t to_load = nbytes;
  uint64_t bytes_read = 0;

  while (to_load) 
  {
    // Only read non-overlapping part of data
    uint64_t to_read = (blocsize - overlap) - current_block_byte;
    if (to_read > to_load) 
      to_read = to_load;

    // Easy case where channels are interleaved or only have 1 channel
    if (time_ordered == false || nchan == 1)
    {
      memcpy(buffer + bytes_read, dat + current_block_byte, to_read);
      bytes_read += to_read;
      current_block_byte += to_read;
    }

    // More complicated case where channel blocks are time-ordered
    else
    {
      // TODO implement this.
      throw Error (InvalidState, "dsp::GUPPIBlockFile::load_bytes",
          "time-order read not implemented yet.");
    }

    // Get next block if necessary
    if (current_block_byte == blocsize - overlap)
    {
      // load_next_block will return 0 if no more data.
      int rv = load_next_block();
      if (rv==0) 
      {
        //eof = true; // TODO need this?
        break;
      }
      current_block_byte = 0;
    }

    to_load -= to_read;

  }

  return nbytes - to_load;

  // Example transpose code:
#if 0 
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

}
