#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "dsp/CPSR2File.h"
#include "dsp/CPSR2_Observation.h"
#include "Error.h"

#include "cpsr2_header.h"
#include "yamasaki_verify.h"
#include "genutil.h"

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

void dsp::CPSR2File::open (const char* filename)
{  
  if (get_header (cpsr2_header, filename) < 0)
    throw Error (FailedCall, "dsp::CPSR2File::open",
		 "get_header(%s) failed", filename);
  
  CPSR2_Observation data (cpsr2_header);

  if (yamasaki_verify (filename, data.offset_bytes, CPSR2_HEADER_SIZE) < 0)
    throw Error (FailedCall, "dsp::CPSR2File::open",
		 "yamasaki_verify(%s) failed", filename);

  info = data;

  // re-open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::CPSR2File::open", 
		 "open(%s) failed", filename);

  struct stat buf;
  if (fstat (fd, &buf) < 0)
    throw Error (FailedSys, "dsp::CPSR2File::open", 
		 "fstat(%s) failed", filename);

  if (buf.st_size < CPSR2_HEADER_SIZE)
    throw Error (InvalidState, "dsp::CPSR2File::open", 
		 "file size=%d < CPSR2 header size=%d",
		 buf.st_size, CPSR2_HEADER_SIZE);

  uint64 total_bytes = buf.st_size - CPSR2_HEADER_SIZE;

  info.set_ndat (info.nsamples (total_bytes));

  // set the number of bytes in header attribute
  header_bytes = CPSR2_HEADER_SIZE;

  // set the file pointers
  reset();

  if (verbose)
    cerr << "CPSR2File::open exit" << endl;
}

