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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

dsp::PdevFile::PdevFile (const char* filename,const char* headername)
  : File ("Pdev")
{
  startfile = 0;
  endfile = 0;
  curfile = 0;
  total_bytes = 0;
  basename[0] = '\0';
  file_bytes.resize(0);
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

void dsp::PdevFile::check_file_set ()
{
  if (verbose)
    cerr << "dsp::PdevFile::check_file_set()" << endl;
  bool more_files = true;
  unsigned ifile = startfile;
  int tmp_fd = 0;
  total_bytes = 0;
  file_bytes.resize(0);
  while (more_files) 
  {
    char tmp_file[256];
    sprintf(tmp_file, "%s.%5.5d.pdev", basename, ifile);
    if (verbose)
      cerr << "dsp::PdevFile::check_file_set testing '" 
        << tmp_file << "'" << endl;
    tmp_fd = ::open(tmp_file, O_RDONLY);
    if (tmp_fd < 0) 
    {
      // All done with files
      more_files = false;
      endfile = ifile - 1;
    }
    else
    {
      // Get size of file to figure out total amount of data
      struct stat buf;
      if (fstat (fd, &buf) < 0)
        throw Error (FailedSys, "dsp::PdevFile::check_file_set", 
            "fstat(%s) failed", tmp_file);
      total_bytes += buf.st_size;
      ::close(tmp_fd);
      file_bytes.push_back(buf.st_size);
      ifile++;
    }
  }
}

int64_t dsp::PdevFile::fstat_file_ndat (uint64_t tailer_bytes)
{
  // This should only be called after check_file_set has been run.
  return info.get_nsamples (total_bytes - PDEV_HEADER_BYTES);
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
  header_bytes = PDEV_HEADER_BYTES;

  // Get the base file name
  if (ascii_header_get (header, "DATAFILE", "%s", basename) < 0)
    throw Error (InvalidParam, "dsp::PdevFile::open_file", 
            "Missing DATAFILE keyword");

  // Get the file with timestamp
  // Assume it is number 0 if not given.
  if (ascii_header_get (header, "STARTFILE", "%d", &startfile) < 0) 
  {
    cerr << "dsp::PdevFile::open_file no STARTFILE given, assuming 0" 
      << endl;
    startfile = 0;
  }

  // Open the initial file to get the header
  char datafile[256];
  curfile = startfile;
  sprintf(datafile, "%s.%5.5d.pdev", basename, startfile);
  fd = ::open(datafile, O_RDONLY);
  if (fd<0)
    throw Error (FailedSys, "dsp::PdevFile::open_file",
            "open(%s) failed", datafile);

  int rv = read (fd, rawhdr, PDEV_HEADER_BYTES);
  if (rv < 0)
    throw Error (FailedSys, "dsp::PdevFile::open_file",
        "Error reading header bytes");

  // TODO check that the header looks reasonable

  MJD epoch ((time_t) rawhdr[12]);
  info.set_start_time(epoch);
  if (verbose)
    cerr << "dsp:PdevFile::open_file start time = " << epoch << endl;

  // Determine total number of files, and total size
  check_file_set();
  set_total_samples();

}

int64_t dsp::PdevFile::load_bytes (unsigned char *buffer, uint64_t nbytes)
{

  if (verbose)
    cerr << "dsp::PdevFile::load_bytes nbytes=" << nbytes << endl;

  // Make sure header_bytes is appropriate for the current file, 
  // header only exists in the first file.
  if (curfile != startfile) 
    header_bytes = 0;

  // Try to load all the bytes from the currently open file. 
  int64_t got_bytes = read (fd, buffer, size_t(nbytes));
  if (got_bytes < 0) 
    throw Error (FailedSys, "dsp::PdevFile::load_bytes", 
        "read(fd) failed");

  // If we got all requested data, or if that was the last file in 
  // the set, we're done.  Also set end_of_data if we've reached
  // the end.
  if (got_bytes == nbytes)
    return got_bytes;
  if (curfile == endfile && got_bytes != nbytes)
  { 
    end_of_data = true;
    return got_bytes;
  }

  // If we still need more data, open the next file.
  File::close();
  curfile++;
  char datafile[256];
  sprintf(datafile, "%s.%5.5d.pdev", basename, curfile);
  end_of_data = false;
  if (verbose)
    cerr << "dsp::PdevFile::load_bytes opening '" << datafile << "'" << endl;
  fd = ::open(datafile, O_RDONLY);
  if (fd<0)
    throw Error (FailedSys, "dsp::PdevFile::load_bytes",
        "open(%s) failed", datafile);

  // be adventurous and recursively call load_bytes here...
  int64_t need_bytes = nbytes - got_bytes;
  got_bytes += load_bytes (buffer + got_bytes, need_bytes);
  return got_bytes;

}

int64_t dsp::PdevFile::seek_bytes (uint64_t bytes)
{

  // Include header in byte count
  bytes += PDEV_HEADER_BYTES;

  // Can't go past the end
  if (bytes >= total_bytes)
  {
    end_of_data = true;
    File::close();
    return total_bytes - PDEV_HEADER_BYTES;
  }

  end_of_data = false;

  // Figure out which file has the right spot
  uint64_t cum_bytes;
  unsigned ifile;
  for (ifile=startfile; ifile<=endfile; ifile++) 
  {
    if ((cum_bytes + file_bytes[ifile-startfile]) > bytes)
      break;
    cum_bytes += file_bytes[ifile-startfile];
  }

  // Close current file, open new file
  File::close();
  curfile = ifile;
  char datafile[256];
  sprintf(datafile, "%s.%5.5d.pdev", basename, curfile);
  if (verbose)
    cerr << "dsp::PdevFile::seek_bytes opening '" << datafile << "'" << endl;
  fd = ::open(datafile, O_RDONLY);
  if (fd<0)
    throw Error (FailedSys, "dsp::PdevFile::seek_bytes",
        "open(%s) failed", datafile);

  // Seek to spot in file
  int64_t rv = lseek(fd, bytes - cum_bytes, SEEK_SET);
  if (rv < 0)
    throw Error (FailedSys, "dsp::PdevFile::seek_bytes", "lseek error");

  return rv - PDEV_HEADER_BYTES;

}
