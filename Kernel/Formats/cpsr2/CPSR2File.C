/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CPSR2File.h"
#include "dsp/CPSR2_Observation.h"
#include "Error.h"

#include "ascii_header.h"
#include "cpsr2_header.h"
#include "yamasaki_verify.h"
#include "dirutil.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

using namespace std;

bool dsp::CPSR2File::want_to_yamasaki_verify = true;

dsp::CPSR2File::~CPSR2File(){}

dsp::CPSR2File::CPSR2File (const char* filename)
  : File ("CPSR2")
{
  if (filename) 
    open (filename);
}

string dsp::CPSR2File::get_prefix () const
{
  return prefix;
}

int dsp::CPSR2File::get_header (char* cpsr2_header, const char* filename)
{
  int fd = ::open (filename, O_RDONLY);

  if (fd < 0) {
    if (verbose)
      fprintf (stderr, "CPSR2File::get_header - failed open(%s): %s", 
	       filename, strerror(errno));
    return -1;
  }

  int retval = read (fd, cpsr2_header, CPSR2_HEADER_SIZE);

  ::close (fd);    

  if (retval < CPSR2_HEADER_SIZE) {
    if (verbose)
      fprintf (stderr, "CPSR2File::get_header - failed read: %s",
	       strerror(errno));

    return -1;
  }

  return 0;
}

static char cpsr2_header [CPSR2_HEADER_SIZE];

bool dsp::CPSR2File::is_valid (const char* filename) const
{
  if (get_header (cpsr2_header, filename) < 0)
    return false;

  // verify that the buffer read contains a valid CPSR2 header
  float version;
  if (ascii_header_get (cpsr2_header, "CPSR2_HEADER_VERSION",
			"%f", &version) < 0)
    return false;

  return true;
}

void dsp::CPSR2File::open_file (const char* filename)
{  
  if (get_header (cpsr2_header, filename) < 0)
    throw Error (FailedCall, "dsp::CPSR2File::open_file",
		 "get_header(%s) failed", filename);
  
  CPSR2_Observation* data = new CPSR2_Observation (cpsr2_header);
  prefix = data->prefix;

  info = data;
  if (get_info()->get_state() == Signal::Intensity)
    throw Error (InvalidState, "dsp::CPSR2File::open_file",
                 "SimpleFB no longer supported");

  header_bytes = CPSR2_HEADER_SIZE;

  if( want_to_yamasaki_verify && filesize(filename) > 50 * 1024 * 1024 )
    if (yamasaki_verify (filename, data->get_offset_bytes(),
			 CPSR2_HEADER_SIZE) < 0)
      throw Error (FailedCall, "dsp::CPSR2File::open_file",
		   "yamasaki_verify(%s) failed", filename);
   
  // re-open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::CPSR2File::open_file()", 
		 "open(%s) failed", filename);
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "CPSR2File::open exit" << endl;
}

//! Pads gaps in data
int64_t dsp::CPSR2File::pad_bytes(unsigned char* buffer, int64_t bytes){
  if( get_info()->get_nbit() != 2 )
    throw Error(InvalidState,"dsp::CPSR2File::pad_bytes()",
		"Can only pad if nbit=2.  nbit=%d",get_info()->get_nbit());

  register const unsigned char val = 255;
  for( int64_t i=0; i<bytes; ++i)
    buffer[i] = val;
  
  return bytes;
}
