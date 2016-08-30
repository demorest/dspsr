/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/DummyFile.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

#include "Error.h"
#include "FilePtr.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

dsp::DummyFile::DummyFile (const char* filename,const char* headername)
  : File ("Dummy")
{
}

dsp::DummyFile::~DummyFile ( )
{
}

bool dsp::DummyFile::is_valid (const char* filename) const
{
  FILE *ptr = fopen(filename, "r");
  if (ptr==NULL) 
    return false;

  char first[32];
  fgets(first, 32, ptr);
  fclose(ptr);
  first[5] = '\0';
  char expect[32] = "DUMMY";
  if (strcmp(first,expect)==0)
    return true;

  if (verbose) 
    cerr << "dsp::DummyFile::is_valid first line != DUMMY" << endl;

  return false;
}

void dsp::DummyFile::open_file (const char* filename)
{
  FILE *ptr = fopen(filename, "r");
  char header[4096];
  fread(header, sizeof(char), 4096, ptr);
  fclose(ptr);
  
  // Read obs info from ASCII file
  info = new ASCIIObservation(header);

  if (ascii_header_get (header, "RESOLUTION", "%u", &resolution) < 0)
    resolution = 1;

  // the resolution is the _byte_ resolution; convert to _sample_ resolution
  if (verbose)
    cerr << "dsp::DummyFile::open_file byte_resolution=" << resolution << endl;
  resolution = info->get_nsamples (resolution);
  if (verbose)
    cerr << "dsp::DummyFile::open_file sample_resolution=" << resolution << endl;
  if (resolution == 0)
    resolution = 1;

}

void dsp::DummyFile::close ()
{
}

int64_t dsp::DummyFile::load_bytes (unsigned char *buffer, uint64_t bytes)
{
  if (verbose)
    cerr << "DummyFile::load_bytes nbytes=" << bytes << endl;
  memset (buffer, 0, bytes);
  return bytes;
}

int64_t dsp::DummyFile::seek_bytes (uint64_t bytes)
{
  if (verbose)
    cerr << "DummyFile::seek_bytes " << bytes << endl;
  return bytes;
}

void dsp::DummyFile::set_total_samples()
{
  if (verbose)
    cerr << "dsp::DummyFile::set_total_samples ndat=" << get_info()->get_ndat() << endl;
}
