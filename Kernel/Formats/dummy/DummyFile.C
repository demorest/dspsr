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
#if GENERATE_BIT_PATTERN
  data = NULL;
  data_size = 0;
  total_bytes = 0;
#endif
  max_bytes = 0;
}

dsp::DummyFile::~DummyFile ( )
{
#if GENERATE_BIT_PATTERN
  if (data!=NULL) free(data);
#endif
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
  char header[1024];
  fread(header, sizeof(char), 1024, ptr);
  fclose(ptr);
  
  // Read obs info from ASCII file
  info = ASCIIObservation(header);
  get_info()->set_machine("Dummy");	

  // Read total amount of data from ascii file
  int max_data_mb = 0;
  if (ascii_header_get(header, "MAX_DATA_MB", "%d", &max_data_mb) >=0)
    max_bytes = max_data_mb * (1L<<20);
}

void dsp::DummyFile::close ()
{
#if GENERATE_BIT_PATTERN
  if (data!=NULL) free(data);
  data = NULL;
#endif
}

int64_t dsp::DummyFile::load_bytes (unsigned char *buffer, uint64_t bytes)
{
  //cerr << "DummyFile::load_bytes " << bytes << endl;
  memset (buffer, 0, bytes);

#if GENERATE_BIT_PATTERN
  if (bytes > max_bytes-total_bytes) { bytes = max_bytes-total_bytes; }
  data = (unsigned char *)realloc(data, bytes); // Maybe not thread-safe...??
  data_size = bytes;
  buffer = data;
  total_bytes += bytes;
#endif

  return bytes;
}

int64_t dsp::DummyFile::seek_bytes(uint64_t bytes)
{
  //cerr << "DummyFile::seek_bytes " << bytes << endl;
  return bytes;
}

void dsp::DummyFile::set_total_samples()
{
  if (max_bytes)
    get_info()->set_ndat( get_info()->get_nsamples(max_bytes) );
}
