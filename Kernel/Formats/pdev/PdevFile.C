/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PdevFile.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

#include "Error.h"
#include "FilePtr.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

dsp::PdevFile::PdevFile (const char* filename,const char* headername)
  : File ("Pdev")
{
  curfile = 0;
  basename[0] = '\0';
}

dsp::PdevFile::~PdevFile ( )
{
}

bool dsp::PdevFile::is_valid (const char* filename) const
{

  FILE *ptr = fopen(filename, "r");
  if (ptr==NULL) 
    return false;

  char header[4096];
  fread(header, sizeof(char), 4096, ptr);
  fclose(ptr);

  // Check for INSTRUMENT = Mock in header
  char inst[64];
  if ( ascii_header_get (header, "INSTRUMENT", "%s", inst) < 0 )
  {
    if (verbose)
      cerr << "dsp::PdevFile::is_valid no INSTRUMENT line" << endl;
    return false;
  }
  if ( std::string(inst) != "Mock" )
  {
    if (verbose)
      cerr << "dsp::PdevFile::is_valid INSTRUMENT is not 'Mock'" << endl;
    return false;
  }

  return true;

}

void dsp::PdevFile::open_file (const char* filename)
{
  FILE *ptr = fopen(filename, "r");
  char header[4096];
  fread(header, sizeof(char), 4096, ptr);
  fclose(ptr);
  
  // Read obs info from ASCII file
  // Start time info is in data file 0, so doesn't 
  // need to be duplicated in header.  Maybe make this
  // so that the header could override the data file value?
  ASCIIObservation info_tmp;
  info_tmp.set_required("UTC_START", false);
  info_tmp.set_required("OBS_OFFSET", false);
  info_tmp.load(header);
  info = info_tmp;

  // Only true for file 0
  header_bytes = 1024;

  // Get the base file name
  if (ascii_header_get (header, "DATAFILE", "%s", basename) < 0)
    throw Error (InvalidParam, "dsp::PdevFile::open_file", 
            "Missing DATAFILE keyword");

  // TODO figure out how to treat the whole multi-file data set as one.
  // Will require reimplementing at least:
  //   load_bytes()
  //   seek_bytes()
  //   fstat_file_ndat()
  // Also need to figure out what to set header_bytes to when file 0
  // is not active anymore.

  // Open the file
  char datafile[256];
  sprintf(datafile, "%s.%5.5d.pdev", basename, curfile);
  fd = ::open(datafile, O_RDONLY);
  if (fd<0)
    throw Error (FailedSys, "dsp::PdevFile::open_file",
            "open(%s) failed", datafile);

  int rv = read (fd, rawhdr, (size_t)1024);
  if (rv < 0)
    throw Error (FailedSys, "dsp::PdevFile::open_file",
        "Error reading header bytes");

  MJD epoch ((time_t) rawhdr[12]);
  info.set_start_time(epoch);
  if (verbose)
    cerr << "dsp:PdevFile::open_file start time = " << epoch << endl;
}
