#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "CPSR2File.h"
#include "CPSR2_Observation.h"
#include "cpsr2_header.h"
#include "yamasaki_verify.h"
#include "genutil.h"

void dsp::CPSR2File::open (const char* filename)
{
  header_bytes = CPSR2_HEADER_SIZE;

  fd = std::open (filename, O_RDONLY);
  if (fd < 0)
    throw_str ("CPSR2File::open - failed open(%s): %s", 
	       filename, strerror(errno));

  char cpsr2_header [CPSR2_HEADER_SIZE];

  int retval = read (fd, cpsr2_header, CPSR2_HEADER_SIZE);

  // close the file in case things go wrong
  std::close (fd);    

  if (retval < CPSR2_HEADER_SIZE)
    throw_str ("CPSR2File::open - failed read: %s", strerror(errno));
  
  CPSR2_Observation data (cpsr2_header);

  if (yamasaki_verify (filename, data.offset_bytes, CPSR2_HEADER_SIZE) < 0)
    throw_str ("cpsr2_Construct: YAMASAKI verification failed");

  info = data;

  // re-open the file
  fd = std::open (filename, O_RDONLY);
  if (fd < 0)
    throw_str ("CPSR2File::open - failed open(%s): %s", 
	       filename, strerror(errno));

}

