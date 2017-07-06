/***************************************************************************
 *
 *   Copyright (C) 2016 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mark5bFile.h"
#include "Error.h"

#include "coord.h"
#include "strutil.h"	
#include "ascii_header.h"

#include <mark5access.h>

#include <memory>
#include <stdio.h>

using namespace std;

dsp::Mark5bFile::Mark5bFile (const char* filename,const char* headername)
  : BlockFile ("Mark5b")
{
  stream = 0;
}

dsp::Mark5bFile::~Mark5bFile ( )
{

}

bool dsp::Mark5bFile::is_valid (const char* filename) const
{
  string headername = filename;
  headername += ".hdr";

  FILE* fptr = fopen (headername.c_str(), "r");
  if( !fptr )
  {
    if (verbose)
      cerr << "Mark5bFile: no hdr file (" << headername << ")" << endl;
    return false;
  }

  auto_ptr<char> header( new char[1024]);
  fread (header.get(), sizeof(char),1024, fptr);
  fclose (fptr);

  char dummy_format[64];
  if (ascii_header_get (header.get(), "FORMAT", "%d", &dummy_format) < 0)
    return false;
	
  return true;
}

void dsp::Mark5bFile::open_file (const char* filename)
{	
  // FIRST Get some vital information from the header file.
  string headername = filename;
  headername += ".hdr";

  FILE *ftext = fopen (headername.c_str(), "r");
	
  if (!ftext) 
    throw Error (FailedSys,"dsp::Mark5bFile",
		 "Cannot open header file " + headername);
	
  char header[1024];
  fread (header, sizeof(char), 1024, ftext);
  fclose (ftext);

  // ///////////////////////////////////////////////////////////////
  //  FORMAT
  //
  char format[64];
  if (ascii_header_get (header,"FORMAT","%s",&format) < 0)
   throw Error (InvalidParam, "Mark5bFile::open_file", 
		 "failed read FORMAT");
	
  cerr << "FORMAT = " << format << endl;

  /* From the mark5access library documentation:

     3.3.1 struct mark5_format_generic* 
                  new_mark5_format_from_string(const char *formatname)

     A function to create a (struct mark5_format_generic) representing
     one of the built-in formats.  The string pointed to by
     "formatname" should be of the form: FORMAT-Mbps-nChannels-nBits.
     Examples for the three formats currently built into mark5acces
     include: "VLBA1_4-256-4-2", "MKIV1_2-128-8-2",
     "Mark5B-1024-16-2".  Note that the string is case insensitive.
     Also note here that in the case of VLBA and Mark4 (MKIV) the
     fanout is built into the FORMAT portion of "formatname".
  */

  struct mark5_format_generic* m5format = 0;
  m5format = new_mark5_format_generic_from_string (format);
  if (!m5format)
    throw Error (FailedCall, "Mark5bFile::open_file",
		 "failed new_mark5_format_generic_from_string (%s)", format);

  fd = 0;

  struct mark5_stream_generic* m5file = 0;
  m5file = new_mark5_stream_file (filename, 0);
  if (!m5file)
    throw Error (FailedCall, "Mark5bFile::open_file",
		 "failed new_mark5_stream_file (%s)", filename);


  struct mark5_stream* m5stream = new_mark5_stream (m5file,m5format);

  stream = m5stream;
  
  // instruct the loader to only take gulps of samplegranularity samples
  Input::resolution = m5stream->samplegranularity;

  int refmjd = 0;
  if (ascii_header_get (header,"REFMJD","%d",&refmjd) < 0)
   throw Error (InvalidParam, "Mark5bFile::open_file", 
		 "failed read REFMJD");

  cerr << "REFMJD " << refmjd << endl;
  m5stream->mjd += refmjd;

  cerr << "MJD = " << m5stream->mjd << endl;
  cerr << "SEC = " << m5stream->sec << endl;

  get_info()->set_start_time( MJD(m5stream->mjd, m5stream->sec, 0) );

  // ///////////////////////////////////////////////////////////////
  // TELESCOPE
  //
	
  char hdrstr[256];
  if (ascii_header_get (header,"TELESCOPE","%s",hdrstr) <0)
    throw Error (InvalidParam, "Mark5bFile::open_file",
		 "failed read TELESCOPE");

  /* user must specify a telescope whose name is recognised or the telescope
     code */

  get_info()->set_telescope (hdrstr);
	
  // ///////////////////////////////////////////////////////////////	
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
    throw Error (InvalidParam, "Mark5bFile::open_file",
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
    throw Error (InvalidParam, "Mark5bFile::open_file",
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
    throw Error (InvalidParam, "Mark5bFile::open_file",
		 "failed read BW");

  get_info()->set_bandwidth (bw);

  double Mega_samples_per_second = m5stream->Mbps / m5stream->nbit;

  double npol = round( (bw * 2) / Mega_samples_per_second );
  cerr << "NPOL=" << npol << endl;
  
  cerr << "NCHAN = " << m5stream->nchan / npol << endl;
  get_info()->set_nchan( m5stream->nchan / npol ); 

  cerr << "NBIT = " << m5stream->nbit << endl;
  get_info()->set_nbit ( m5stream->nbit );
  
  cerr << "SAMPRATE = " << m5stream->samprate << endl;
  get_info()->set_rate ( m5stream->samprate );

  get_info()->set_npol(npol);

  // ///////////////////////////////////////////////////////////////	
  // NDIM  --- whether the data are Nyquist or Quadrature sampled
  //
  // MARK5 data are Nyquist sampled

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
  get_info()->set_machine("Mark5b");	
}

int64_t dsp::Mark5bFile::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose) cerr << "Mark5bFile::load_bytes nbytes =" << bytes << endl;

  if (verbose) 
    cerr << "Mark5bFile::load_bytes leave it to MARK5_stream_get_data" << endl;
  return bytes;
}

int64_t dsp::Mark5bFile::seek_bytes (uint64_t nbytes)
{
  if (verbose)
    cerr << "Mark5bFile::seek_bytes nbytes=" << nbytes << endl;

  if (nbytes != 0)
    throw Error (InvalidState, "Mark5bFile::seek_bytes", "unsupported");

  return nbytes;
}


void dsp::Mark5bFile::reopen ()
{
  throw Error (InvalidState, "Mark5bFile::reopen", "unsupported");
}

