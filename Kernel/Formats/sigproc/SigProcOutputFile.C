/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SigProcOutputFile.h"
#include "dsp/SigProcObservation.h"
#include "FilePtr.h"

#include <fcntl.h>

using namespace std;

dsp::SigProcOutputFile::SigProcOutputFile (const char* filename) 
  : OutputFile ("SigProcOutputFile")
{
  if (filename) 
    output_filename = filename;
}


//! Get the extension to be added to the end of new filenames
std::string dsp::SigProcOutputFile::get_extension () const
{
  return ".fil";
}

void dsp::SigProcOutputFile::write_header ()
{
  FilePtr fptr = tmpfile();
  if (!fptr)
    throw Error (FailedSys, "dsp::SigProcOutputFile::write_header",
		 "error tmpfile");

  SigProcObservation data;

  data.copy( get_input() );
  data.unload( fptr );

  long bytes = ftell( fptr );
  rewind( fptr );

  vector<char> buffer (bytes);

  size_t loaded = fread (&(buffer[0]), 1, bytes, fptr);
  if (loaded != (size_t) bytes)
    throw Error (FailedSys, "dsp::SigProcOutputFile::write_header",
		 "error fread %ld bytes", bytes);

  unload_bytes (&(buffer[0]), bytes);
}


