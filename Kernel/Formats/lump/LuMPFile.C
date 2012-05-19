/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "dsp/LuMPFile.h"
#include "dsp/LuMPObservation.h"
#include "ascii_header.h"
#include "Error.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::LuMPFile::LuMPFile (const char* filename) : File ("LuMP")
{
    if (verbose)
        cerr << "dsp::LuMPFile::LuMPFile used" << endl;
    if (filename) {
        open (filename);
    }
}

string dsp::LuMPFile::get_header (const char* filename)
{
  std::string line;
  std::ifstream input (filename);

  if (!input)
    return line;

  std::getline (input, line, '\0');
  //if (verbose)
  //    std::cerr << "dsp::LuMPFile::get_header header " << line << endl;

  return line;
}

bool dsp::LuMPFile::is_valid (const char* filename) const
{
  string header = get_header (filename);

  if (header.empty())
  {
    if (verbose)
      cerr << "dsp::LuMPFile::is_valid empty header" << endl;
    return false;
  }

  // verify that the buffer read contains a valid LuMP header
  float version;
  if (ascii_header_get (header.c_str(), "LUMP_VERSION", "%f", &version) < 0)
  {
    if (verbose)
      cerr << "dsp::LuMPFile::is_valid LUMP_VERSION not defined" << endl;
    return false;
  }

  return true;

  LuMPObservation data (filename);
  return true;
}

void dsp::LuMPFile::open_file (const char* filename)
{
  string header = dsp::LuMPFile::get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::LuMPFile::open_file",
		 "get_header(%s) failed", filename);
  
  lump_info = new LuMPObservation (header.c_str());
  info = *lump_info;

  unsigned hdr_size;
  if (ascii_header_get (header.c_str(), "HDR_SIZE", "%u", &hdr_size) < 0)
    throw Error (FailedCall, "dsp::LuMPFile::open_file",
		 "ascii_header_get(HDR_SIZE) failed");
  if (verbose)
      cerr << "header size = " << hdr_size << " bytes" << endl;
  header_bytes = hdr_size;

  // open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::LuMPFile::open_file()", 
		 "open(%s) failed", filename);
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / info.get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "LuMPFile::open exit" << endl;
}
