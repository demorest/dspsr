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

#include "fitshead.h"
#include "fitshead_utils.h"

using namespace std;

dsp::GUPPIFile::GUPPIFile (const char* filename)
  : GUPPIBlockFile ("GUPPI")
{
  pos0 = 0;
}

dsp::GUPPIFile::~GUPPIFile ( )
{
  // Free things.
  if (hdr != NULL) free(hdr);
  if (dat != NULL) free(dat);
}

// Read header starting from current spot in file.  Puts header info
// into hdr buf, and returns total number of keys found.  If no
// "END" is found, frees hdr and return 0.
// TODO could rewrite this using the fitshead.h stuff now, but 
// this already exists..
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

  rv = hget(hdr_tmp, "BLOCSIZE", &itmp);
  if (rv<0) {
    if (verbose)
      cerr << "dsp::GUPPIFile coulnd't find BLOCSIZE keyword." << endl;
    return false;
  }

  rv = hget(hdr_tmp, "PKTIDX", &itmp);
  if (rv<0) {
    if (verbose)
      cerr << "dsp::GUPPIFile coulnd't find PKTIDX keyword." << endl;
    return false;
  }

  // Everything passed, it's probably a GUPPI file.
  return true;
}

void dsp::GUPPIFile::open_file (const char* filename)
{

  // Open file
  fd = ::open(filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::GUPPIFile::open_file",
        "open(%s) failed", filename);

  // Read in header
  int hdr_keys = get_header(fd, &hdr);
  if (hdr_keys<=0) 
    throw Error (InvalidState, "dsp::GUPPIFile::open_file",
        "Error parsing block0 header.");

  // Parse header into info struct
  parse_header();

  // Save the first block size, then look ahead to the next block.
  // If the block size is different, we will ignore the first one.
  uint64_t blocsize0 = blocsize;
  int hdr_keys0 = hdr_keys;
  int rv = lseek(fd, blocsize, SEEK_CUR);
  if (rv<0) 
    throw Error (FailedSys, "dsp::GUPPIFile::open_file",
        "lseek(block1) failed");
  hdr_keys = get_header(fd, &hdr);
  if (hdr_keys<=0) 
    throw Error (InvalidState, "dsp::GUPPIFile::open_file",
        "Error parsing block1 header.");
  parse_header();
  if (blocsize != blocsize0) {
    // Set pos0 to point to start of 2nd data block
    pos0 = blocsize0 + 80*hdr_keys0;
  }

  // Now rewind to the right spot, load 1st block and re-parse header
  seek_bytes(0);
  parse_header();

  // Figure out total size
  // dfac thing accounts for some weirdness in definition of 
  // OVERLAP for real-sampled TDOM data..
  struct stat buf;
  rv = fstat(fd, &buf);
  if (rv < 0)
    throw Error (FailedSys, "dsp::GUPPIFile::open_file",
        "fstat(%s) failed", filename);
  uint64_t full_block_size = blocsize + 80*hdr_keys;
  uint64_t nblocks = (buf.st_size-pos0) / full_block_size;
  unsigned int dfac = get_info()->get_ndim()==1 ? 2 : 1;
  get_info()->set_ndat( get_info()->get_nsamples(nblocks*blocsize) - dfac*overlap*nblocks );

}

int dsp::GUPPIFile::load_next_block ()
{
  // Assume we're at the right spot in the file, is start of
  // next header.  For now, we will ignore header params
  // and assume nothing is changing or skipped.

  // This could also be 0 at EOF.
  int nkeys = get_header(fd, &hdr);
  if (nkeys==0) 
    return 0;
    //throw Error (InvalidState, "dsp::GUPPIFile::load_next_block",
    //    "Error loading next header");

  // Make sure memory is allocated
  if (dat==NULL)
    dat = (unsigned char *)malloc(blocsize);

  // Read the data
  size_t nbytes = read(fd, dat, blocsize);
  if (nbytes < 0)
    throw Error (FailedSys, "dsp::GUPPIFile::load_next_block", "read");

  if (nbytes != blocsize) return 0;

  return 1;
}

int64_t dsp::GUPPIFile::seek_bytes (uint64_t bytes)
{
  if (bytes==0)
  {
    int rv = lseek(fd, pos0, SEEK_SET);
    if (rv<0) 
      throw Error (FailedSys, "dsp::GUPPIFile::seek_bytes",
          "lseek(0) failed");
    load_next_block();
    
    current_block_byte = 0;
    return 0;
  }
  else
    throw Error (InvalidState, "dsp::GUPPIFile::seek_bytes",
        "seek(%lld) not implemented yet", bytes);
}

void dsp::GUPPIFile::close ()
{
  // Free up the memory buffer if file has been closed
  if (dat != NULL) 
  {
    free(dat);
    dat = NULL;
  }
  // Call the standard close routine
  File::close();
}
