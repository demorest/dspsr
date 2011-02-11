/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/VDIFFile.h"
#include "vdifio.h"
#include "Error.h"

#include "coord.h"
#include "strutil.h"	
#include "ascii_header.h"

#include <iomanip>

#include <time.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

dsp::VDIFFile::VDIFFile (const char* filename,const char* headername)
  : BlockFile ("VDIF")
{
  stream = 0;
}

dsp::VDIFFile::~VDIFFile ( )
{

}

bool dsp::VDIFFile::is_valid (const char* filename) const
{

  FILE* fptr = fopen (filename, "r");
  if( !fptr ) {
      if (verbose) 
          cerr << "VDIFFile: Error opening file." << endl;
    return false;
  }

  // Read one header
  char rawhdr[VDIF_HEADER_BYTES];
  size_t rv = fread(rawhdr, sizeof(char), VDIF_HEADER_BYTES, fptr);
  fclose(fptr);
  if (rv != VDIF_HEADER_BYTES) {
      if (verbose) 
          cerr << "VDIFFile: Error reading header." << endl;
    return false;
  }

  // See if some basic values make sense
  int nbytes = getVDIFFrameBytes(rawhdr);
  if (nbytes<0 || nbytes>MAX_VDIF_FRAME_BYTES) {
      if (verbose) 
          cerr << "VDIFFFile: Frame bytes = " << nbytes << endl;
      return false;
  }

  int mjd = getVDIFFrameMJD(rawhdr);
  if (mjd<30000 || mjd>70000) {
      if (verbose) 
          cerr << "VDIFFFile: MJD = " << mjd << endl;
      return false;
  }

  int nchan = getVDIFNumChannels(rawhdr);
  if (nchan<0 || nchan>nbytes*8) {
      if (verbose) 
          cerr << "VDIFFFile: nchan = " << nchan << endl;
      return false;
  }
	
  // Everything looks ok
  return true;
}

void dsp::VDIFFile::open_file (const char* filename)
{	

  FILE *fptr = fopen (filename, "r");
  fd = ::open(filename, O_RDONLY);
  if (fd<0) 
      throw Error (FailedSys, "dsp::VDIFFile::open_file",
              "open(%s) failed", filename);
	
  // Read first header
  char rawhdr[VDIF_HEADER_BYTES];
  size_t rv = read(fd, rawhdr, VDIF_HEADER_BYTES);
  if (rv != VDIF_HEADER_BYTES) 
      throw Error (FailedSys, "VDIFFile::open_file",
              "Error reading first header");
  lseek(fd, 0, SEEK_SET);

  // Get basic params

  int nbit = getVDIFBitsPerSample(rawhdr);
  if (verbose) cerr << "NBIT = " << nbit << endl;
  info.set_nbit (nbit);

  int mjd = getVDIFFrameMJD(rawhdr);
  int sec = getVDIFFrameSecond(rawhdr);
  int fn = getVDIFFrameNumber(rawhdr);
  double frames_per_sec = 8000.0; // XXX hack
  if (verbose) cerr << "MJD = " << mjd << endl;
  if (verbose) cerr << "Sec = " << sec << endl;
  info.set_start_time( MJD(mjd,sec,(double)fn/frames_per_sec) );

  int nchan = getVDIFNumChannels(rawhdr);
  if (verbose) cerr << "NCHAN = " << nchan << endl;
  info.set_nchan( nchan ); 

  // TODO where do these come from..
  float mbps = 64;
  info.set_npol(1);
  info.set_state (Signal::Nyquist); // XXX ??
  info.set_rate ( 1e6 * mbps / info.get_npol() / nchan / nbit );
  info.set_bandwidth ( -1.0 * mbps / info.get_npol() / nbit / 2 );
  info.set_telescope ("vla");
  //info.set_source (hdrstr);
  //info.set_coordinates(coords);
  info.set_centre_frequency (1658.0 + 8.0);
	  
  int nbyte = getVDIFFrameBytes(rawhdr);
  header_bytes = 0;
  block_bytes = nbyte;
  block_header_bytes = VDIF_HEADER_BYTES; // XXX what about "legacy" mode

  set_total_samples();

  // XXX Why?
  //header_bytes = block_header_bytes = 0;

  //
  // call this only after setting frequency and telescope
  //

  string prefix="tmp";    // what prefix should we assign??
	  
  info.set_mode(stringprintf ("%d-bit mode",info.get_nbit() ) );
  info.set_machine("VDIF");	
}

//int64_t dsp::VDIFFile::seek_bytes (uint64_t nbytes)
//{
//  if (verbose)
//    cerr << "VDIFFile::seek_bytes nbytes=" << nbytes << endl;
//
//  if (nbytes != 0)
//    throw Error (InvalidState, "VDIFFile::seek_bytes", "unsupported");
//
//  return nbytes;
//}


void dsp::VDIFFile::reopen ()
{
  throw Error (InvalidState, "Mark5File::reopen", "unsupported");
}
