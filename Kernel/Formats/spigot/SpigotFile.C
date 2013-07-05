/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/SpigotFile.h"
#include "FITSError.h"

#include <fitsio.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include <memory>

using namespace std;


dsp::SpigotFile::SpigotFile (const char* filename)
  : File ("Spigot")
{
  if (filename)
    open (filename);
}

dsp::SpigotFile::~SpigotFile ()
{

}

bool dsp::SpigotFile::is_valid (const char* filename) const
{
 if (verbose)
    cerr << "dsp::SpigotFile::is_valid " << filename << endl;

  int status = 0;  
  char error [FLEN_ERRMSG];

  fitsfile* fptr = 0;  
  fits_open_file (&fptr, filename, READONLY, &status);
  
  if (status != 0 && verbose) {
    fits_get_errstatus (status, error);
    cerr << "dsp::SpigotFile::is_valid fits_open_file: " << error << endl;
  }

  // do not return comments in fits_read_key
  char* comment = 0;
  auto_ptr<char> tempstr (new char [FLEN_VALUE]);

  if (!status)
    fits_read_key (fptr, TSTRING, "INSTRUME", tempstr.get(), comment, &status);

  if (status != 0 && verbose) {
    fits_get_errstatus (status, error);
    cerr << "dsp::SpigotFile::is_valid FITS fits_read_key: " << error << endl;
  }

  fits_close_file (fptr, &status);

  if (status != 0)
    return false;

  return true;

}

void dsp::SpigotFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::SpigotFile::open_file " << filename << endl;

  int status = 0;
  fitsfile* fptr = 0;

  fits_open_file (&fptr, filename, READONLY, &status);
  
  if (status != 0)
    throw FITSError (status, "dsp::SpigotFile::open_file", 
		     "fits_open_file(%s)", filename);

  // do not return comments in fits_read_key
  char* comment = 0;

  long headsize;
  fits_read_key (fptr, TLONG, "HEADSIZE", &headsize, comment, &status);

  if (status != 0)
    throw FITSError (status, "dsp::SpigotFile::open_file", 
		     "fits_read_key (HEADSIZE)");

  header_bytes = headsize;

  parse (fptr);

  fits_close_file (fptr, &status);

  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::SpigotFile::open", 
		 "failed open(%s)", filename);

  if (verbose)
    cerr << "dsp::SpigotFile::open exit" << endl;
}
  
void dsp::SpigotFile::parse (void* header)
{
  int status = 0;
  fitsfile* fptr = reinterpret_cast<fitsfile*> (header);

  /* Until further notice */
  get_info()->set_basis (Signal::Linear);
  get_info()->set_type  (Signal::Pulsar);

  // do not return comments in fits_read_key
  char* comment = 0;
  auto_ptr<char> tempstr (new char [FLEN_VALUE]);

  //
  // mode
  //

  fits_read_key (fptr, TSTRING, "MODE", tempstr.get(), comment, &status);

  if (status != 0)
    throw FITSError (status, "dsp::SpigotFile::parse", 
		     "fits_read_key (MODE)");

  get_info()->set_mode( tempstr.get() );

  //
  // state and npol
  //

  fits_read_key (fptr, TSTRING, "SUMPOL", tempstr.get(), comment, &status);

  string sumpol = tempstr.get();

  if (sumpol == "T") {
    get_info()->set_state (Signal::Intensity);
    get_info()->set_npol (1);
  }
  else {
    cerr << "dsp::SpigotFile::parse not sure if SUMPOL==" << sumpol
	 << " means XXYY, LLRR, or IQUV, or what" << endl;
    throw Error (InvalidState, "dsp::SpigotFile::parse",
		 "unknown SUMPOL");
  }

  //
  // nchan
  //

  int nchan;
  fits_read_key (fptr, TINT, "NLAGS", &nchan, comment, &status);
  get_info()->set_nchan (nchan);
  
  /* for now data is always real-valued */
  get_info()->set_ndim (1);

  //
  // nbit
  //

  int nbit;
  fits_read_key (fptr, TINT, "BITS", &nbit, comment, &status);
  get_info()->set_nbit (nbit);

  if (verbose)
    cerr << "dsp::SpigotFile::parse " << get_info()->get_nbyte() << " bytes/sample" 
         << endl;

  // Always Greenbank for now
  get_info()->set_telescope ( "Greenbank" );

  //
  // source
  //

  fits_read_key (fptr, TSTRING, "OBJECT", tempstr.get(), comment, &status);

  if (status != 0)
    throw FITSError (status, "dsp::SpigotFile::parse", 
		     "fits_read_key (MODE)");

  get_info()->set_source (tempstr.get());

  if (verbose)
    cerr << "dsp::SpigotFile::parse source=" << get_info()->get_source() << endl;

  //
  // coordinates
  //

  double ra;
  double dec;

  fits_read_key (fptr, TDOUBLE, "RA", &ra, comment, &status);
  fits_read_key (fptr, TDOUBLE, "DEC", &dec, comment, &status);

  sky_coord position;
  
  position.ra().setDegrees( ra );
  position.dec().setDegrees( dec );
  
  get_info()->set_coordinates( position );
  
  //
  // centre_frequency
  //
  
  double cfreq;
  fits_read_key (fptr, TDOUBLE, "CENTFREQ", &cfreq, comment, &status);
  get_info()->set_centre_frequency( cfreq );
  
  //
  // bandwidth
  //

  fits_read_key (fptr, TSTRING, "UPPERSB", tempstr.get(), comment, &status);

  if (status != 0)
    throw FITSError (status, "dsp::SpigotFile::parse", 
		     "fits_read_key (UPPERSB)");

  string uppersb = tempstr.get();
  double sign = 1.0;

  if (uppersb == "T")
    sign = 1.0;
  else if (uppersb == "F")
    sign = -1.0;
  else
    throw Error (InvalidState, "dsp::SpigotFile::parse",
		 "unknown UPPERSB=" + uppersb);

  double bw;
  fits_read_key (fptr, TDOUBLE, "SAMP-BW", &bw, comment, &status);
  get_info()->set_bandwidth( sign * bw );

  //
  // start_time
  //
  long time_obs;
  fits_read_key (fptr, TLONG, "SEC-OBS", &time_obs, comment, &status);

  MJD mjd_obs( (time_t) time_obs );

  if (verbose)
    cerr << "dsp::SpigotFile::parse start = " << mjd_obs << endl;

  get_info()->set_start_time( mjd_obs );

  //
  // sampling rate
  //

  double tsamp;
  fits_read_key (fptr, TDOUBLE, "TSAMP", &tsamp, comment, &status);
  get_info()->set_rate (1e6/tsamp);  // tsamp in microseconds

  //
  // ndat
  //
  long spectra;
  fits_read_key (fptr, TLONG, "SPECTRA", &spectra, comment, &status);
  get_info()->set_ndat( spectra );

  get_info()->set_scale (1.0);
  
  get_info()->set_swap (false);
  
  get_info()->set_dc_centred (false);

  fits_read_key (fptr, TSTRING, "BASENAME", tempstr.get(), comment, &status);
  get_info()->set_identifier (tempstr.get());
  
  get_info()->set_machine ("Spigot");
  
  get_info()->set_dispersion_measure (0);
}

