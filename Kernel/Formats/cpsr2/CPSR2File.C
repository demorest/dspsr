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

bool dsp::CPSR2File::want_to_yamasaki_verify = true;

dsp::CPSR2File::CPSR2File (const char* filename)
  : File ("CPSR2")
{
  if (filename) 
    open (filename);
}

//! Return a pointer to an possibly identical instance of a CPSR2File
dsp::CPSR2File* dsp::CPSR2File::clone(bool identical){
  CPSR2File* copy = new CPSR2File;

  //if( identical )
  //copy->operator=( *this );

  return copy;
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

void dsp::CPSR2File::open_file (const char* filename,PseudoFile* _info)
{  
  if( _info ){
    info = *_info;
    header_bytes = CPSR2_HEADER_SIZE;
  }
  else{
    if (get_header (cpsr2_header, filename) < 0)
      throw Error (FailedCall, "dsp::CPSR2File::open_file()",
		   "get_header(%s) failed", filename);
  
    CPSR2_Observation data (cpsr2_header);
    
    if( want_to_yamasaki_verify )
      if (yamasaki_verify (filename, data.offset_bytes, CPSR2_HEADER_SIZE) < 0)
	throw Error (FailedCall, "dsp::CPSR2File::open_file()",
		     "yamasaki_verify(%s) failed", filename);

    info = data; 

    header_bytes = CPSR2_HEADER_SIZE;
  }

  // re-open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::CPSR2File::open_file()", 
		 "open(%s) failed", filename);
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / info.get_nbit();
  if (resolution == 0)
    resolution = 1;

  // set the file pointers
  reset();

  if (verbose)
    cerr << "CPSR2File::open exit" << endl;
}
















