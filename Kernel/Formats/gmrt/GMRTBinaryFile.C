/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTBinaryFile.h"
#include "dsp/ASCIIObservation.h"
#include "Error.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::GMRTBinaryFile::GMRTBinaryFile (const char* filename) : File ("GMRT")
{
  if (filename) 
    open (filename);
}


string dsp::GMRTBinaryFile::get_header (const char* filename)
{
  string hdr_filename = filename + string(".hdr");

  ifstream input ( hdr_filename.c_str() );

  string line;

  if (!input)
  {
    if (verbose)
      std::cerr << "dsp::GMRTBinaryFile::get_header"
	" could not open '" << hdr_filename << "'" << endl;
    return line;
  }

  getline (input, line, '\0');

  return line;
}

bool dsp::GMRTBinaryFile::is_valid (const char* filename) const
{
  string header = get_header (filename);

  if (header.empty())
  {
    if (verbose)
      cerr << "dsp::GMRTBinaryFile::is_valid empty header" << endl;
    return false;
  }

  try
  {
    ASCIIObservation test;
    test.load( header.c_str() );
    return true;
  }
  catch (Error& error)
    {
      if (verbose)
	cerr << "dsp::GMRTBinaryFile::is_valid "
	     << error.get_message() << endl;
    }

  return false;
}

void dsp::GMRTBinaryFile::open_file (const char* filename)
{  
  string header = get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::GMRTBinaryFile::open_file",
		 "get_header(%s) failed", filename);
  
  info = new ASCIIObservation (header.c_str());

  header_bytes = 0;

  resolution = get_info()->get_nsamples (1);
  if (resolution == 0)
    resolution = 1;

  open_fd (filename);
}
