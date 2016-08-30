/***************************************************************************
 *
 *   Copyright (C) 2005-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/Mark5File.h"
#include "dsp/Mark5TwoBitCorrection.h"
#include "vlba_stream.h"
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

dsp::Mark5File::Mark5File (const char* filename,const char* headername)
  : BlockFile ("Mark5")
{
  stream = 0;
}

dsp::Mark5File::~Mark5File ( )
{

}

bool dsp::Mark5File::is_valid (const char* filename) const
{
  string headername = filename;
  headername += ".hdr";

  FILE* fptr = fopen (headername.c_str(), "r");
  if( !fptr ) {
      if (verbose) cerr << "Mark5File: no hdr file (" << headername << ")" << endl;
    return false;
  }

  auto_ptr<char> header( new char[1024]);
  fread (header.get(), sizeof(char),1024, fptr);
  fclose (fptr);

  int dummy_fanout = 0;
  if (ascii_header_get (header.get(), "FANOUT", "%d", &dummy_fanout) < 0)
    return false;
	
  return true;
}

void dsp::Mark5File::open_file (const char* filename)
{	
  // FIRST Get some vital information from the header file.
  string headername = filename;
  headername += ".hdr";

  FILE *ftext = fopen (headername.c_str(), "r");
	
  if (!ftext) 
    throw Error (FailedSys,"dsp::Mark5File",
		 "Cannot open header file " + headername);
	
  char header[1024];
  fread (header, sizeof(char), 1024, ftext);
  fclose (ftext);

  // ///////////////////////////////////////////////////////////////
  //  NBIT
  //
  int nbit = 0;
  if (ascii_header_get (header,"NBIT","%d",&nbit) < 0)
   throw Error (InvalidParam, "Mark5File::open_file", 
		 "failed read NBIT");
	
  cerr << "NBIT = " << nbit << endl;
  get_info()->set_nbit (nbit);


  // ///////////////////////////////////////////////////////////////
  //  FANOUT
  //
  int fanout = 0;
  if (ascii_header_get (header,"FANOUT","%d",&fanout) < 0)
   throw Error (InvalidParam, "Mark5File::open_file", 
		 "failed read FANOUT");
	
  cerr << "FANOUT = " << fanout << endl;

  struct VLBA_stream* vlba_stream = 0;

  stream = vlba_stream = VLBA_stream_open (filename, nbit, fanout, 0);

  if (!stream)
    throw Error (InvalidParam, "Mark5File::open_file",
		 "failed VLBA_stream_open");

  fd = 0;

  // instruct the loader to only take gulps in 32/16 lots of nbits
  // necessary since Mk5 files are written in 64-/32-bit words
  cerr << "TRACKS = " << vlba_stream->tracks << endl;
  Input::resolution = vlba_stream->tracks / nbit;  

  // The factor of 2 should only apply for dual-pol data.
  cerr << "NCHAN = " << vlba_stream->nchan / 2 << endl;
  get_info()->set_nchan( vlba_stream->nchan / 2 ); 

  cerr << "SAMPRATE = " << vlba_stream->samprate << endl;
  get_info()->set_rate ( vlba_stream->samprate );

  int refmjd = 0;
  if (ascii_header_get (header,"REFMJD","%d",&refmjd) < 0)
   throw Error (InvalidParam, "Mark5File::open_file", 
		 "failed read REFMJD");

  cerr << "REFMJD " << refmjd << endl;
  vlba_stream->mjd += refmjd;

  cerr << "MJD = " << vlba_stream->mjd << endl;
  cerr << "SEC = " << vlba_stream->sec << endl;

  get_info()->set_start_time( MJD(vlba_stream->mjd, vlba_stream->sec, 0) );

  // ///////////////////////////////////////////////////////////////
  // TELESCOPE
  //
	
  char hdrstr[256];
  if (ascii_header_get (header,"TELESCOPE","%s",hdrstr) <0)
    throw Error (InvalidParam, "Mark5File::open_file",
		 "failed read TELESCOPE");

  /* user must specify a telescope whose name is recognised or the telescope
     code */

  get_info()->set_telescope (hdrstr);
	
  // ///////////////////////////////////////////////////////////////	
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
    throw Error (InvalidParam, "Mark5File::open_file",
		 "failed read SOURCE");

  get_info()->set_source (hdrstr);

  // ///////////////////////////////////////////////////////////////	
  // COORDINATES
  //
  int rv=0;
  bool got_coords = true;
  double ra=0.0, dec=0.0;
  if (got_coords && ascii_header_get (header, "RA", "%s", hdrstr) >= 0) {
    cerr << "RASTR = '" << hdrstr << "'" << endl;
    rv = str2dec2(&ra, hdrstr);
    ra *= 360.0/24.0;
    if (rv==0) 
      got_coords = true;
    else 
      got_coords = false;
  } else 
    got_coords = false;
  if (got_coords && ascii_header_get (header, "DEC", "%s", hdrstr) >= 0) {
    cerr << "DECSTR = '" << hdrstr << "'" << endl;
    if (str2dec2(&dec, hdrstr)==0)
      got_coords = true;
    else 
      got_coords = false;
  } else
    got_coords = false;

  if (got_coords) {
    cerr << "RA = " << ra*12.0/M_PI << endl;
    cerr << "DEC = " << dec*180.0/M_PI << endl;
    sky_coord coords;
    coords.setRadians(ra,dec);
    get_info()->set_coordinates(coords);
  }


  // ///////////////////////////////////////////////////////////////
  // FREQ
  //
  // Note that we assign the CENTRE frequency, not the edge of the band
  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    throw Error (InvalidParam, "Mark5File::open_file",
		 "failed read FREQ");

  get_info()->set_centre_frequency (freq);
	
  //
  // WvS - flag means that even number of channels are result of FFT
  // get_info()->set_dc_centred(true);

  // ///////////////////////////////////////////////////////////////
  // BW
  //
  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    throw Error (InvalidParam, "Mark5File::open_file",
		 "failed read BW");

  get_info()->set_bandwidth (bw);
	
  // ///////////////////////////////////////////////////////////////
  // NPOL
  //	
  //  -- generalise this later
	
  get_info()->set_npol(2);    // read in both polns at once

  // ///////////////////////////////////////////////////////////////	
  // NDIM  --- whether the data are Nyquist or Quadrature sampled
  //
  // VLBA data are Nyquist sampled

  get_info()->set_state (Signal::Nyquist);
	  
  // ///////////////////////////////////////////////////////////////
  // NDAT
  // Compute using BlockFile::fstat_file_ndat
  //
  header_bytes = 0;
  block_bytes = FRAMESIZE;
  block_header_bytes = FRAMESIZE - PAYLOADSIZE;

  set_total_samples();

  header_bytes = block_header_bytes = 0;

  //
  // call this only after setting frequency and telescope
  //

  string prefix="tmp";    // what prefix should we assign??
	  
  get_info()->set_mode(stringprintf ("%d-bit mode",get_info()->get_nbit() ) );
  get_info()->set_machine("Mark5");	
}

extern "C" int next_frame (struct VLBA_stream *vs);

/*! Uses Walter's next_frame to take care of the modbits business, then
 copies the result from the VLBA_stream::frame buffer into the buffer
 argument. */
int64_t dsp::Mark5File::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose) cerr << "Mark5File::load_bytes nbytes =" << bytes << endl;

  if (!dsp::Mark5TwoBitCorrection::can_do( get_info() )) {
    if (verbose) 
      cerr << "Mark5File::load_bytes leave it to VLBA_stream_get_data" << endl;
    return bytes;
  }

  struct VLBA_stream* vlba_stream = (struct VLBA_stream*) stream;

  unsigned char* from = (unsigned char*) vlba_stream->payload;
  unsigned char* to = buffer;

  unsigned offset_word = vlba_stream->read_position;
  unsigned step = vlba_stream->fanout * unsigned(get_info()->get_nbyte()+0.5);

  unsigned bytes_read = 0;

#if _DEBUG
  cerr << "PAYLOADSIZE=" << PAYLOADSIZE << endl;
#endif

  while (bytes_read < bytes) {

    if (offset_word >= PAYLOADSIZE) {

      int retval = next_frame(vlba_stream);

      // cerr << "retval=" << retval << endl;

      if (retval < 0)
      {
	set_eod (true);
	break;
      }

      from = (unsigned char*) vlba_stream->payload;
      offset_word = 0;

    }

#if _DEBUG
    cerr << "bytes=" << bytes << " bytes_read=" << bytes_read
	 << "\n frame=" << vlba_stream->framenum
	 << " offset=" << offset_word << " step=" << step 
	 << "\n firstvalid=" << vlba_stream->firstvalid
	 << " lastvalid=" << vlba_stream->lastvalid << endl;
#endif

    unsigned max_bytes = bytes - bytes_read;
    unsigned done_bytes = 0;

    unsigned invalid_bytes = 0;

    if (offset_word < (unsigned) vlba_stream->firstvalid) {
      invalid_bytes = (vlba_stream->firstvalid - offset_word) * step;
      offset_word = vlba_stream->firstvalid;
    }
    else if (offset_word > (unsigned) vlba_stream->lastvalid) {
      invalid_bytes = (PAYLOADSIZE - offset_word) * step;
      offset_word = PAYLOADSIZE;
    }

    if (invalid_bytes) {

      if (invalid_bytes > max_bytes)
	invalid_bytes = max_bytes;

#if _DEBUG
      cerr << "  invalid bytes = " << invalid_bytes << endl;
#endif
      memset (to, 0, invalid_bytes);
      done_bytes = invalid_bytes;

    }

    else {

      unsigned last_word = PAYLOADSIZE;
      if (vlba_stream->lastvalid+1 < PAYLOADSIZE)
	last_word = vlba_stream->lastvalid+1;

      unsigned copy_bytes = (last_word - offset_word) * step;

      if (copy_bytes > max_bytes)
	copy_bytes = max_bytes;

#if _DEBUG
      cerr << "  copy bytes = " << copy_bytes << endl;
#endif
      memcpy (to, from, copy_bytes);
      done_bytes = copy_bytes;

    }

    to += done_bytes;
    from += done_bytes;
    bytes_read += done_bytes;
    offset_word += done_bytes / step;

    vlba_stream->read_position = offset_word;

  }

  return bytes_read;
}

int64_t dsp::Mark5File::seek_bytes (uint64_t nbytes)
{
  if (verbose)
    cerr << "Mark5File::seek_bytes nbytes=" << nbytes << endl;

  if (nbytes != 0)
    throw Error (InvalidState, "Mark5File::seek_bytes", "unsupported");

  return nbytes;
}


void dsp::Mark5File::reopen ()
{
  throw Error (InvalidState, "Mark5File::reopen", "unsupported");
}
