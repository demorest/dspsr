/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PuMa2File.h"
#include "dsp/PuMa2_Observation.h"
#include "ascii_header.h"
#include "Error.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

bool dsp::PuMa2File::want_to_check_bocf = true;

int puma2_check_bocf (const char* filename, uint64_t offset_bytes,
		       uint64_t search_offset)
{
  cerr << "puma2_check_bocf: WARNING not implemented" << endl;
  return 0;
}


dsp::PuMa2File::PuMa2File (const char* filename) : File ("PuMa2")
{
  if (filename) 
    open (filename);
}


string dsp::PuMa2File::get_header (const char* filename)
{
  std::string line;
  std::ifstream input (filename);

  if (!input)
    return line;

  std::getline (input, line, '\0');

  return line;
}

bool dsp::PuMa2File::is_valid (const char* filename) const
{
  string header = get_header (filename);

  if (header.empty())
  {
    if (verbose)
      cerr << "dsp::PuMa2File::is_valid empty header" << endl;
    return false;
  }

  // verify that the buffer read contains a valid PuMa2 header
  float version;
  if (ascii_header_get (header.c_str(), "HDR_VERSION", "%f", &version) < 0)
  {
    if (verbose)
      cerr << "dsp::PuMa2File::is_valid HDR_VERSION not defined" << endl;
    return false;
  }

  return true;
}

void dsp::PuMa2File::open_file (const char* filename)
{  
  string header = get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::PuMa2File::open_file",
		 "get_header(%s) failed", filename);
  
  PuMa2_Observation* data = new PuMa2_Observation (header.c_str());
  info = data;

  unsigned hdr_size;
  if (ascii_header_get (header.c_str(), "HDR_SIZE", "%u", &hdr_size) < 0)
    throw Error (FailedCall, "dsp::PuMa2File::open_file",
		 "ascii_header_get(HDR_SIZE) failed");
  cerr << "header size = " << hdr_size << " bytes" << endl;
  header_bytes = hdr_size;

  if( want_to_check_bocf &&
      puma2_check_bocf (filename, data->get_offset_bytes(), hdr_size) < 0 )
    throw Error (FailedCall, "dsp::PuMa2File::open_file",
		 "check_bocf(%s) failed", filename);
   
  // open the file
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::PuMa2File::open_file()", 
		 "open(%s) failed", filename);
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "PuMa2File::open exit" << endl;
}


