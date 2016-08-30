/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/GUPPIBuffer.h"

#include "Error.h"

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "fitshead.h"
#include "fitshead_utils.h"

#include "ascii_header.h"

#include "guppi_databuf.h"
#include "guppi_error.h"

using namespace std;

dsp::GUPPIBuffer::GUPPIBuffer (const char* filename)
  : GUPPIBlockFile ("GUPPIBuffer")
{
  databuf_id = 0;
  databuf = NULL;
  curblock = -1;
  got_stt_valid = false;
}

dsp::GUPPIBuffer::~GUPPIBuffer ( )
{
  // detach from databuf if necessary
  if (databuf != NULL) guppi_databuf_detach(databuf);
}

bool dsp::GUPPIBuffer::is_valid (const char* filename) const
{

  // Read a simple ascii file to get databuf number

  FILE *f = fopen(filename, "r");
  if (f==NULL)
    return false;

  char header[4096];
  header[0] = '\0';
  fread(header, sizeof(char), 4096, f);
  fclose(f);

  // check for INSTRUMENT = guppi_daq
  char inst[64];
  if ( ascii_header_get(header, "INSTRUMENT", "%s", inst) < 0 )
  {
    if (verbose)
      cerr << "dsp::GUPPIBuffer::is_valid no INSTRUMENT line" << endl;
    return false;
  }
  if (std::string(inst) != "guppi_daq") 
  {
    if (verbose)
      cerr << "dsp::GUPPIBuffer::is_valid INSTRUMENT is not 'guppi_daq'" 
        << endl;
    return false;
  }

  return true;
}

void dsp::GUPPIBuffer::open_file (const char* filename)
{

  // Open the ascii header file
  FILE *f = fopen(filename, "r");
  if (f==NULL)
    throw Error (FailedSys, "dsp::GUPPIBuffer::open_file",
        "fopen(%s) returned NULL", filename);
  char header[4096];
  fread(header, sizeof(char), 4096, f);
  fclose(f);

  // Get databuf id number
  if (ascii_header_get(header, "DATABUF", "%d", &databuf_id) < 0)
    throw Error (InvalidParam, "dsp::GUPPIBuffer::open_file",
        "Missing DATABUF keyword");

  // Connect to databuf
  databuf = guppi_databuf_attach(databuf_id);
  if (databuf==NULL)
    throw Error (InvalidState, "dsp::GUPPIBuffer::open_file", 
        "Error connecting to databuf %d", databuf_id);

  // Wait for first block
  load_next_block();

  // Once we have it, parse the header get_info()->
  parse_header();

}

int dsp::GUPPIBuffer::load_next_block ()
{

  do {

    // Done with current block
    if (curblock >= 0) 
      guppi_databuf_set_free(databuf, curblock);

    // Wait for next block to be filled
    curblock = (curblock + 1) % databuf->n_block;
    bool waiting = true;
    while (waiting) {
      if (verbose)
        cerr << "dsp::GUPPIBuffer::load_next_block waiting(" 
          << curblock << ")" << endl;
      int rv = guppi_databuf_wait_filled(databuf, curblock);
      if (rv==0) 
        waiting = false;
      else if (rv==GUPPI_TIMEOUT)
        waiting = true;
      else
        throw Error (InvalidState, "dsp::GUPPIBuffer::load_next_block",
            "guppi_databuf_wait_filled returned %d", rv);
    }

    hdr = guppi_databuf_header(databuf, curblock);
    dat = (unsigned char *)guppi_databuf_data(databuf, curblock);

    int stt_valid=0;
    hget(hdr, "STTVALID", &stt_valid);
    if (stt_valid) got_stt_valid = true;

  } while (!got_stt_valid);

  return 1;
}

int64_t dsp::GUPPIBuffer::seek_bytes (uint64_t bytes)
{
  // TODO figure out how to do this.. seeking could be ok as
  // long as it doesn't go too far backwards.
  if (bytes==0)
  {
    current_block_byte = 0;
    return 0;
  }
  else
    throw Error (InvalidState, "dsp::GUPPIBuffer::seek_bytes",
        "seek(%lld) not implemented yet", bytes);
}
