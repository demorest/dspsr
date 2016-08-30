/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/LWAFile.h"
#include "dsp/ASCIIObservation.h"
#include "Error.h"

#include "coord.h"
#include "strutil.h"	
#include "ascii_header.h"
#include "machine_endian.h"

#include <iomanip>

#include <time.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

dsp::LWAFile::LWAFile (const char* filename,const char* headername)
  : BlockFile ("LWA")
{
  stream = 0;
  datafile[0] = '\0';
}

dsp::LWAFile::~LWAFile ( )
{
}

bool dsp::LWAFile::is_valid (const char* filename) const
{

  // Open the header file, check for INSTRUMENT=LWA
  // TODO use a different keyword?
  FILE *fptr = fopen(filename, "r");
  if (!fptr) 
  {
    if (verbose)
      cerr << "dsp::LWAFile::is_valid Error opening file." << endl;
    return false;
  }

  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  char inst[64];
  if ( ascii_header_get(header, "INSTRUMENT", "%s", inst) < 0 )
  {
    if (verbose)
      cerr << "dsp::LWAFile::is_valid no INSTRUMENT line" << endl;
    return false;
  }
  if ( std::string(inst) != "LWA" )
  {
    if (verbose)
      cerr << "dsp::LWAFile::is_valid INSTRUMENT != 'LWA'" << endl;
    return false;
  }

  // TODO check for DATAFILE line?

  // Everything looks ok
  return true;
}

void dsp::LWAFile::open_file (const char* filename)
{	

  // This is the header file
  FILE *fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::LWAFile::open_file",
        "fopen(%s) failed", filename);

  // Read the header
  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  // Get the data file
  if (ascii_header_get (header, "DATAFILE", "%s", datafile) < 0)
    throw Error (InvalidParam, "dsp::LWAFile::open_file", 
        "Missing DATAFILE keyword");

  // Parse the standard ASCII info.  Timestamps are in LWA packets
  // so not required.  
  ASCIIObservation* info_tmp = new ASCIIObservation;
  info = info_tmp;

  info_tmp->set_required("UTC_START", false);
  info_tmp->set_required("OBS_OFFSET", false);
  info_tmp->set_required("NPOL", true); 
  info_tmp->set_required("NBIT", false); //XXX always 4-bit?
  info_tmp->set_required("NDIM", false);
  info_tmp->set_required("NCHAN", false); //XXX always 1
  info_tmp->set_required("TSAMP", false);
  info_tmp->set_required("BW", false);
  info_tmp->load(header);

  fd = ::open(datafile, O_RDONLY);
  if (fd<0) 
      throw Error (FailedSys, "dsp::LWAFile::open_file",
              "open(%s) failed", datafile);
	
  // Read first header
  char rawhdr_bytes[LWA_HEADER_BYTES];
  size_t rv = read(fd, rawhdr_bytes, LWA_HEADER_BYTES);
  if (rv != LWA_HEADER_BYTES) 
      throw Error (FailedSys, "LWAFile::open_file",
              "Error reading first header");
  lseek(fd, 0, SEEK_SET);

  // TODO check for magic bytes 0xDEC0DE5C ?

  // Get basic params

  get_info()->set_nbit(4);
  get_info()->set_ndim(2);
  get_info()->set_state(Signal::Analytic);
  get_info()->set_nchan(1); // Always 1

  header_bytes = 0;
  block_header_bytes = LWA_HEADER_BYTES;
  block_bytes = LWA_HEADER_BYTES 
                   + get_info()->get_npol() * LWA_DATA_BYTES;

  // Decimation rate
  const double lwa_base_bw = 196.0;
  uint16_t dec_fac = *(uint16_t *)(&rawhdr_bytes[12]);
  FromBigEndian(dec_fac);
  double bw = lwa_base_bw / (double)dec_fac;
  if (verbose) cerr << "LWAFile::open_file dec_fac=" << dec_fac 
    << " bw=" << bw << endl;
  get_info()->set_bandwidth(bw);

  get_info()->set_rate( (double) get_info()->get_bandwidth() * 1e6 
      / (double) get_info()->get_nchan() 
      * (get_info()->get_state() == Signal::Nyquist ? 2.0 : 1.0));
  if (verbose) cerr << "LWAFile::open_file rate = " << get_info()->get_rate() << endl;

  uint16_t timeoff = *((uint16_t *)(&rawhdr_bytes[14]));
  FromBigEndian(timeoff);
  uint64_t timetag = *((uint64_t *)(&rawhdr_bytes[16]));
  FromBigEndian(timetag);
  if (verbose) cerr << "LWAFile::timetag = " << timetag << endl;
  long double timetag_sec = (long double)timetag / (lwa_base_bw*1e6);

  if (verbose) cerr << "LWAFile::timetag_sec = " << setprecision(20) 
    << timetag_sec << endl;
  MJD mjd;
  mjd = MJD((time_t)timetag_sec);
  int imjd = mjd.intday();
  int smjd = mjd.get_secs();
  if (verbose) cerr << "LWAFile::imjd = " << imjd << endl;
  if (verbose) cerr << "LWAFile::smjd = " << smjd << endl;
  double t_offset = timetag_sec - floorl(timetag_sec);
  if (verbose) cerr << "LWAFile::t_offset = " << setprecision(20) << t_offset << endl;
  get_info()->set_start_time( MJD(imjd,smjd,t_offset) );
  if (verbose) cerr << "LWAFile::start_time = " << setprecision(20) <<
    get_info()->get_start_time().in_days() << endl;

  // Figures out how much data is in file based on header sizes, etc.
  set_total_samples();
}

//int64_t dsp::LWAFile::seek_bytes (uint64_t nbytes)
//{
//  if (verbose)
//    cerr << "LWAFile::seek_bytes nbytes=" << nbytes << endl;
//
//  if (nbytes != 0)
//    throw Error (InvalidState, "LWAFile::seek_bytes", "unsupported");
//
//  return nbytes;
//}


void dsp::LWAFile::reopen ()
{
  throw Error (InvalidState, "dsp::LWAFile::reopen", "unsupported");
}
