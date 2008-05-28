/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAFile.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

#include "Error.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::DADAFile::DADAFile (const char* filename) : File ("DADA")
{
  if (filename) 
    open (filename);
}

string dsp::DADAFile::get_header (const char* filename)
{
  std::string line;
  std::ifstream input (filename);

  if (!input)
    return line;

  std::getline (input, line, '\0');

  return line;
}

bool dsp::DADAFile::is_valid (const char* filename) const
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

void dsp::DADAFile::open_file (const char* filename)
{  
  string header = get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::DADAFile::open_file",
		 "get_header(%s) failed", filename);
  
  ASCIIObservation data (header.c_str());
  info = data;

  if (ascii_header_get (header.c_str(), "HDR_SIZE", "%u", &header_bytes) < 0)
    throw Error (FailedCall, "dsp::DADAFile::open_file",
		 "ascii_header_get(HDR_SIZE) failed");

  if (ascii_header_get (header.c_str(), "RESOLUTION", "%u", &resolution) < 0)
    resolution = 1;

  // the resolution is the _byte_ resolution; convert to _sample_ resolution
  resolution = info.get_nsamples (resolution);
  if (resolution == 0)
    resolution = 1;

  // open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::DADAFile::open_file()", 
		 "open(%s) failed", filename);

  if (verbose)
    cerr << "DADAFile::open exit" << endl;
}


