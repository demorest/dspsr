/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/VDIFFile.h"
#include "dsp/ASCIIObservation.h"
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
  datafile[0] = '\0';
}

dsp::VDIFFile::~VDIFFile ( )
{
}

bool dsp::VDIFFile::is_valid (const char* filename) const
{

  // Open the header file, check for INSTRUMENT=VDIF
  // TODO use a different keyword?
  FILE *fptr = fopen(filename, "r");
  if (!fptr) 
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid Error opening file." << endl;
    return false;
  }

  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  char inst[64];
  if ( ascii_header_get(header, "INSTRUMENT", "%s", inst) < 0 )
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid no INSTRUMENT line" << endl;
    return false;
  }
  if ( std::string(inst) != "VDIF" )
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid INSTRUMENT != 'VDIF'" << endl;
    return false;
  }

  // TODO check for DATAFILE line?

  // Old code below.  Could use to also test datafile.
#if 0 

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
#endif
	
  // Everything looks ok
  return true;
}

void dsp::VDIFFile::open_file (const char* filename)
{	

  // This is the header file
  FILE *fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::VDIFFile::open_file",
        "fopen(%s) failed", filename);

  // Read the header
  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  // Get the data file
  if (ascii_header_get (header, "DATAFILE", "%s", datafile) < 0)
    throw Error (InvalidParam, "dsp::VDIFFile::open_file", 
        "Missing DATAFILE keyword");

  // Parse the standard ASCII info.  Timestamps are in VDIF packets
  // so not required.  Also we'll assume VDIF's "nchan" really gives
  // the number of polns for now, and NCHAN is 1.  NBIT is in VDIF packets.
  // We'll compute TSAMP from the bandwidth.  NDIM (real vs complex sampling)
  // is in VDIF packets but there is no easy fn to parse it yet.
  ASCIIObservation info_tmp;
  info_tmp.set_required("UTC_START", false);
  info_tmp.set_required("OBS_OFFSET", false);
  info_tmp.set_required("NPOL", false);
  info_tmp.set_required("NBIT", false);
  info_tmp.set_required("NCHAN", false);
  info_tmp.set_required("TSAMP", false);
  info_tmp.load(header);
  info = info_tmp;

  fd = ::open(datafile, O_RDONLY);
  if (fd<0) 
      throw Error (FailedSys, "dsp::VDIFFile::open_file",
              "open(%s) failed", datafile);
	
  // Read first header
  char rawhdr[VDIF_HEADER_BYTES];
  size_t rv = read(fd, rawhdr, VDIF_HEADER_BYTES);
  if (rv != VDIF_HEADER_BYTES) 
      throw Error (FailedSys, "VDIFFile::open_file",
              "Error reading first header");
  lseek(fd, 0, SEEK_SET);

  // Get basic params

  int nbit = getVDIFBitsPerSample(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file NBIT = " << nbit << endl;
  info.set_nbit (nbit);

  int nbyte = getVDIFFrameBytes(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file FrameBytes = " << nbyte << endl;
  header_bytes = 0;
  block_bytes = nbyte;
  block_header_bytes = VDIF_HEADER_BYTES; // XXX what about "legacy" mode

  // Each poln shows up as a different channel but this 
  // could also be different freq channels...
  int vdif_nchan = getVDIFNumChannels(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file NCHAN = " << vdif_nchan << endl;
  if ((vdif_nchan<0) || (vdif_nchan>2))
    throw Error (InvalidParam, "dsp::VDIFFile::open_file",
        "Read vdif_nchan=%d, this is currently not supported", vdif_nchan);
  info.set_npol( vdif_nchan );
  info.set_nchan( 1 );
  info.set_rate( (double) info.get_bandwidth() * 1e6 
      / (double) info.get_nchan() 
      * info.get_state() == Signal::Nyquist ? 2.0 : 1.0);

  // Figure frames per sec from bw, pkt size, etc
  //double frames_per_sec = 64000.0;
  int frame_data_size = nbyte - VDIF_HEADER_BYTES;
  double frames_per_sec = info.get_nbit() * info.get_nchan() * info.get_npol()
    * info.get_rate() / 8.0 / (double) frame_data_size;

  int mjd = getVDIFFrameMJD(rawhdr);
  int sec = getVDIFFrameSecond(rawhdr);
  int fn = getVDIFFrameNumber(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file MJD = " << mjd << endl;
  if (verbose) cerr << "VDIFFile::open_file sec = " << sec << endl;
  info.set_start_time( MJD(mjd,sec,(double)fn/frames_per_sec) );

  // XXX old code, should all be handled by ASCII header now
  //float mbps = 512 * info.get_npol();
  //info.set_state (Signal::Nyquist);
  //info.set_rate ( 1e6 * mbps / info.get_npol() / info.get_nchan() / nbit );
  //info.set_bandwidth ( 1.0 * mbps / info.get_npol() / nbit / 2 );
  //info.set_telescope ("vla");
  //info.set_source (hdrstr);
  //info.set_coordinates(coords);
  //info.set_centre_frequency (1658.0 + 8.0);
  //info.set_centre_frequency (1458.0);
	  
  // Figures out how much data is in file based on header sizes, etc.
  set_total_samples();

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
  throw Error (InvalidState, "dsp::VDIFFile::reopen", "unsupported");
}
