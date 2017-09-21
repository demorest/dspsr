/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/DADAFile.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

#include "FilePtr.h"
#include "Error.h"
#include "strutil.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::DADAFile::DADAFile (const char* filename) : File ("DADA")
{
  separate_header_file = false;
  
  if (filename) 
    open (filename);
}

string dsp::DADAFile::get_header (const char* filename) const
{
  FilePtr fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::DADAFile::get_header",
		 "fopen (%s)", filename);

  // default DADA header size
  long hdr_size = 4096;
  vector<char> buffer;
  char* header = 0;

  do
  {
    ::rewind (fptr);

    buffer.resize (hdr_size);
    header = &(buffer[0]);

    if (fread (header, 1, hdr_size, fptr) != hdr_size)
      throw Error (FailedSys, "dsp::DADAFile::get_header",
		   "fread (nbyte=%u)", hdr_size);

    // ensure that text is null-terminated before calling ascii_header_get
    header[ hdr_size-1 ] = '\0';

    /* Get the header size */
    if (ascii_header_get (header, "HDR_SIZE", "%u", &hdr_size) != 1)
      hdr_size = 0;

    /* Ensure that the incoming header fits in the client header buffer */
  }
  while (hdr_size > buffer.size());

  if (hdr_size == 0)
  {
    // search for a matching .hdr file
    string hdr_ext = ".hdr";
    string hdr_fname = replace_extension (filename, hdr_ext);
    FilePtr hdr_ptr = fopen (hdr_fname.c_str(), "r");
    if (!fptr)
    {
      hdr_fname = filename + hdr_ext;
      hdr_ptr = fopen (hdr_fname.c_str(), "r");
    }
    
    if (!hdr_ptr)
      throw Error (InvalidState, "dsp::DADAFile::get_header",
		   "file has no header and no matching header file found");

    if (fseek (hdr_ptr, 0, SEEK_END) < 0)
      throw Error (FailedSys, "dsp::DADAFile::get_header",
		   "could not fseek to end of header file");

    hdr_size = ftell (hdr_ptr);
    if (hdr_size < 0)
      throw Error (FailedSys, "dsp::DADAFile::get_header",
		   "ftell fails at end of header file");

    ::rewind (hdr_ptr);

    buffer.resize (hdr_size);
    header = &(buffer[0]);

    if (fread (header, 1, hdr_size, hdr_ptr) != hdr_size)
      throw Error (FailedSys, "dsp::DADAFile::get_header",
		   "fread (nbyte=%u) from header file", hdr_size);

    // ensure that text is null-terminated before calling ascii_header_get
    header[ hdr_size-1 ] = '\0';
    separate_header_file = true;
  }
  
  if (!header)
    return string();

  return header;
}

bool dsp::DADAFile::is_valid (const char* filename) const try
{
  string header = get_header (filename);

  if (header.empty())
  {
    if (verbose)
      cerr << "dsp::DADAFile::is_valid empty header" << endl;
    return false;
  }

  // verify that the buffer read contains a valid DADA header
  float version;
  if (ascii_header_get (header.c_str(), "HDR_VERSION", "%f", &version) < 0)
  {
    if (verbose)
      cerr << "dsp::DADAFile::is_valid HDR_VERSION not defined" << endl;
    return false;
  }

  // verify that the buffer read contains a valid DADA header
  char instrument[64];
  if (ascii_header_get (header.c_str(), "INSTRUMENT", "%s", instrument) < 0)
  {
    if (verbose)
      cerr << "dsp::DADAFile::is_valid INSTRUMENT not defined" << endl;
    return false;
  }

  return true;
}
 catch (Error& error)
   {
     if (verbose)
       cerr << "dsp::DADAFile::is_valid " << error.get_message() << endl;
     return false;
   }

void dsp::DADAFile::open_file (const char* filename)
{  
  string header = get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::DADAFile::open_file",
		 "get_header(%s) failed", filename);
  
  info = new ASCIIObservation (header.c_str());

  const char* hdr = header.c_str();
  
  if (separate_header_file)
    header_bytes = 0;
  else if (ascii_header_get (hdr, "HDR_SIZE", "%u", &header_bytes) < 0)
    throw Error (FailedCall, "dsp::DADAFile::open_file",
		 "ascii_header_get(HDR_SIZE) failed");

  if (ascii_header_get (hdr, "RESOLUTION", "%u", &resolution) < 0)
    resolution = 1;

  // the resolution is the _byte_ resolution; convert to _sample_ resolution
  resolution = info->get_nsamples (resolution);
  if (resolution == 0)
    resolution = 1;

  open_fd (filename);

  if (verbose)
    cerr << "DADAFile::open exit" << endl;
}


