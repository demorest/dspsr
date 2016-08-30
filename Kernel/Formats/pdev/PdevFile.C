/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PdevFile.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

#include "pdev_aoHdr.h"

#include "Error.h"
#include "FilePtr.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

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

  // Check to see if this is a raw pdev file
  uint32_t *magic = (uint32_t *)header;
  if (*magic == 0xfeffbeef)
  {
    // TODO see if we can parse the filename also?
    return true;
  }
  if (verbose)
    cerr << "dsp::PdevFile::is_valid no magic number, trying ASCII header" 
      << endl;

  // Check for INSTRUMENT = Mock in ASCII header
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
  return get_info()->get_nsamples (total_bytes - PDEV_HEADER_BYTES);
}

void dsp::PdevFile::open_file (const char* filename)
{

  FILE *ptr = fopen(filename, "r");
  char header[4096];
  fread(header, sizeof(char), 4096, ptr);
  fclose(ptr);
  
  // Only true for file 0
  header_bytes = PDEV_HEADER_BYTES;

  // Check for magic number in first 4 bytes
  uint32_t *magic = (uint32_t *)header;
  if (*magic == 0xfeffbeef)
  {
    have_ascii_header = false;

    // Parse filename to get base / filenum.
    // It should end in .NNNNN.pdev
    char fnametmp[256];
    strcpy(fnametmp, filename);
    char *cptr = strrchr(fnametmp, '.');
    if (cptr == NULL)
      throw Error (InvalidState, "dsp::PdevFile::open_file",
          "Error parsing filename (%s)", filename);
    // TODO could check for .pdev here
    *cptr = '\0';
    cptr = strrchr(fnametmp,'.');
    if (cptr == NULL)
      throw Error (InvalidState, "dsp::PdevFile::open_file",
          "Error parsing filename (%s)", filename);
    *cptr = '\0';
    cptr++;
    startfile = atoi(cptr);
    strcpy(basename, fnametmp);
  }

  else 
  {
    have_ascii_header = true;

    // Get the base file name from ASCII header
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

    // Parse standard ASCII header, minus obs start time
    ASCIIObservation* info_tmp = new ASCIIObservation;
    info_tmp->set_required("UTC_START", false);
    info_tmp->set_required("OBS_OFFSET", false);
    info_tmp->load(header);
    info = info_tmp;
  }

  if (verbose)
    cerr << "dsp::PdevFile::open_file basename=" << basename
      << " startfile=" << startfile << endl;

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

  // Check for valid header
  // 1st int should be 0xfeffbeef
  if (rawhdr[0] != 0xfeffbeef)
    throw Error (InvalidState, "dsp::PdevFile::open_file",
        "Magic number 0xfeffbeef is not present.");

  // rawhdr[14] equal 0x12345678 if aoHdr is present
  if (have_ascii_header == false)
  {
    if (rawhdr[14] == 0x12345678)
      parse_aoHdr();
    else
      throw Error (InvalidState, "dsp::PdevFile::open_file",
          "No aoHdr found -- use ASCII header file instead");
  }

  MJD epoch ((time_t) rawhdr[12]);
  get_info()->set_start_time(epoch);
  if (verbose)
    cerr << "dsp:PdevFile::open_file start time = " << epoch << endl;

  // Determine total number of files, and total size
  check_file_set();
  set_total_samples();

}

void dsp::PdevFile::parse_aoHdr ()
{
  if (verbose)
    cerr << "dsp::PdevFile::parse_aoHdr" << endl;

  char *aoHdr_raw = (char *)rawhdr + 240;
  struct pdev_aoHdr *aohdr = (struct pdev_aoHdr *)aoHdr_raw;
  char strtmp[32];

  // TODO make sure endian is ok

  // Check for expected hdrVer
  strncpy(strtmp, aohdr->hdrVer, 4); strtmp[4] = '\0';
  if (strncmp(aohdr->hdrVer,"1.00",4) != 0) 
    throw Error (InvalidParam, "dsp::PdevFile::parse_aoHdr",
        "Unrecognized hdrVer value (%s)", strtmp);

  // This stuff should always be true..
  get_info()->set_telescope("Arecibo");
  get_info()->set_machine("Mock");
  get_info()->set_npol(2);
  get_info()->set_nbit(8);
  get_info()->set_ndim(2);
  get_info()->set_nchan(1);
  get_info()->set_state(Signal::Analytic);

  if (verbose)
    cerr << "dsp::PdevFile::parse_aoHdr bw=" << aohdr->bandWdHz
      << " dir=" << aohdr->bandIncrFreq << endl;
  double bw = aohdr->bandWdHz / 1e6;
  if (aohdr->bandIncrFreq == 0) bw *= -1.0;
  cerr << "pdev: forcing bw inversion! bw was:" << bw;
  bw = bw * -1.0;
  cerr << "Now bw=" << bw << endl;
  get_info()->set_bandwidth(bw);
  get_info()->set_rate(aohdr->bandWdHz);

  get_info()->set_centre_frequency(aohdr->cfrHz / 1e6);

  strncpy(strtmp, aohdr->object, 16); strtmp[16] = '\0';
  get_info()->set_source(strtmp);

  strncpy(strtmp, aohdr->frontEnd, 8); strtmp[8] = '\0';
  get_info()->set_receiver(strtmp);

  sky_coord coords;
  coords.setDegrees(aohdr->raJDeg, aohdr->decJDeg);
  get_info()->set_coordinates(coords);

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
  uint64_t cum_bytes=0;
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
