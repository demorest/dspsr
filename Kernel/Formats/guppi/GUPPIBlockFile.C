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

  header_get_check("NBITS", &itmp);
  info.set_nbit(itmp);

  header_get_check("OBSBW", &ftmp);
  info.set_bandwidth(ftmp);

  header_get_check("OBSFREQ", &ftmp);
  info.set_centre_frequency(ftmp);
 
  header_get_check("OBSNCHAN", &itmp);
  info.set_nchan(itmp);

  header_get_check("NPOL", &itmp);
  if (itmp==1)
    info.set_npol(1);
  else 
    info.set_npol(2);

  // Use packet format flag for this for now...
  header_get_check("PKTFMT", ctmp);
  if (string(ctmp) == "VDIF")
    info.set_state(Signal::Nyquist);
  else
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
  if (verbose)
    cerr << "dsp::GUPPIBlockFile::parse_header ra_str=" 
      << ctmp << " dec_str=" << ctmp2 << endl;
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
// NOTE: Could handle this by setting resolution to only
// allow reads in multiples of the block size.  But this 
// would constrain the allowed block sizes for processing.
int64_t dsp::GUPPIBlockFile::load_bytes (unsigned char *buffer, uint64_t nbytes)
{
  if (verbose) 
    cerr << "dsp::GUPPIBlockFile::load_bytes() nbytes=" << nbytes << endl;

  const unsigned nchan = info.get_nchan();
  const unsigned npol = info.get_npol();
  const unsigned nbit = info.get_nbit();
  const unsigned bytes_per_samp = (2 * npol * nbit) / 8;
  const uint64_t overlap_bytes = overlap * bytes_per_samp * nchan;
  const uint64_t blocsize_per_chan = blocsize / nchan;
  uint64_t to_load = nbytes;
  uint64_t bytes_read = 0;

  while (to_load && !end_of_data) 
  {
    // Only read non-overlapping part of data
    uint64_t to_read = (blocsize - overlap_bytes) - current_block_byte;
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

      uint64_t to_read_per_chan = to_read / nchan;
      uint64_t nsamp_per_chan = to_read_per_chan / bytes_per_samp;
      uint64_t cur_byte = current_block_byte / nchan;

      for (unsigned isamp=0; isamp<nsamp_per_chan; isamp++) {
        for (unsigned ichan=0; ichan<nchan; ichan++) 
        {
          memcpy(buffer + bytes_read, 
              dat + cur_byte + ichan*blocsize_per_chan + isamp*bytes_per_samp,
              bytes_per_samp);
          bytes_read += bytes_per_samp;
        }
      }

      current_block_byte += to_read;

    }

    to_load -= to_read;

    // Get next block if necessary
    if (current_block_byte == blocsize - overlap_bytes)
    {
      // load_next_block will return 0 if no more data.
      int rv = load_next_block();
      if (rv==0) 
      {
        end_of_data = true;
        break;
      }
      current_block_byte = 0;
    }

  }

  return nbytes - to_load;

}
