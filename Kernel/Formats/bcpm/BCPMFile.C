#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "dsp/BCPMFile.h"

dsp::BCPMFile::BCPMFile (const char* filename) : File ("BCPM"){
  if (filename)
    open (filename,0);
}

dsp::BCPMFile::~BCPMFile (){ }

bool dsp::BCPMFile::is_valid (const char* filename,int) const
{
  
}

void dsp::BCPMFile::open_file (const char* filename)
{
  
}
