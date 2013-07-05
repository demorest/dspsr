/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GMRTFilterbankFile.h"
#include "dsp/PrestoObservation.h"

#include "FilePtr.h"

#include <string.h>

using namespace std;

// #define _DEBUG

dsp::GMRTFilterbankFile::GMRTFilterbankFile (const char* filename)
  : File ("GMRTFilterbank")
{
  if (filename)
    open (filename);
}

std::string header_filename (const char* filename)
{
  unsigned length = strlen (filename);
  if (strcmp(filename+length-3, "dat"))
    throw Error (InvalidParam, "GMRTFilterbankFile::header_filename",
		 "%s does not end int 'dat'", filename);

  string output_filename = filename;
  output_filename.replace (length-3, 3, "hdr");
  return output_filename;
}

bool dsp::GMRTFilterbankFile::is_valid (const char* filename) const try
{
  string hdr_filename = header_filename (filename);

  FilePtr fptr = fopen (hdr_filename.c_str(), "r");
  if (!fptr)
  {
    if (verbose)
      cerr << "dsp::GMRTFilterbankFile::is_valid could not open "
	   << hdr_filename << endl;
    return false;
  }

  char first_line [80];
  if (!fgets (first_line, 80, fptr))
  {
    if (verbose)
      cerr << "dsp::GMRTFilterbankFile::is_valid could not read from "
	   << hdr_filename << endl;
    return false;
  }

  if (strncmp (first_line, "# DATA FILE HEADER #", 20))
  {
    if (verbose)
      cerr << "dsp::GMRTFilterbankFile::is_valid unexpected first line of "
	   << hdr_filename << endl;
    return false;
  }

  return true;
}
catch (Error& error)
{
  if (verbose)
    cerr <<"dsp::GMRTFilterbankFile::is_valid "<< error.get_message() <<endl;
  return false;
}

// defined in gmrt.c
extern "C" { int GMRT_hdr_to_inf(char *datfilenm, infodata *idata); }

/*! 
  Loads the Observation information from an GMRTFilterbank-TCI style file.
*/
void dsp::GMRTFilterbankFile::open_file (const char* filename)
{
  string hdr_filename = header_filename (filename);

  FilePtr fptr = fopen (hdr_filename.c_str(), "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::GMRTFilterbankFile::open_file",
		 "fopen (" + hdr_filename + ")");

  open_fd (filename);

  infodata data;
  GMRT_hdr_to_inf (const_cast<char*>(filename), &data);
  
  if (verbose)
    cerr << "dsp::GMRTFilterbankFile::open PRESTO finished" << endl;

  info = new PrestoObservation (&data);
  get_info()->set_basis (Signal::Circular);

  header_bytes = 0;
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "dsp::GMRTFilterbankFile::open return" << endl;
}


