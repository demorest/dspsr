/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "dsp/SigProcFile.h"
#include "dsp/SigProcObservation.h"
#include <fcntl.h>

using namespace std;

dsp::SigProcFile::SigProcFile (const char* filename) : File ("SigProc")
{
  if (filename) 
    open (filename);
}

bool dsp::SigProcFile::is_valid (const char* filename) const
{
  SigProcObservation data (filename);
  return true;
}

void dsp::SigProcFile::open_file (const char* filename)
{
  SigProcObservation* data = new SigProcObservation (filename);

  info = data;
  header_bytes = data->header_bytes;
   
  // open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::SigProcFile::open_file()", 
		 "open(%s) failed", filename);

  if (verbose)
    cerr << "SigProcFile::open exit" << endl;
}


