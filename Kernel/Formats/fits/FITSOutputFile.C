/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/FITSOutputFile.h"
#include "dsp/Observation.h"
#include "dsp/Rescale.h"
#include "FilePtr.h"

#include "FITSArchive.h"
#include "Pulsar/FITSHdrExtension.h"
#include "Pulsar/dspReduction.h"
#include "Pulsar/Telescope.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/Backend.h"

#include <fcntl.h>

using namespace std;

dsp::FITSOutputFile::FITSOutputFile (unsigned bits, const char* filename) 
  : OutputFile ("FITSOutputFile")
{
  if (filename) 
    output_filename = filename;
  cerr << "dsp::FITSOutputFile constructor " << output_filename << endl;
  nchan = 0;
  npol = 0;
  nsblk = 4096;
  nbblk = 0;
  nbit = bits;
  offset = 0;
  written = 0;
  isub = 0;
  fptr = NULL;
  dat_wts = NULL;
  dat_offs = NULL;
  dat_scl = NULL;
  dat_freq = NULL;
}

dsp::FITSOutputFile::~FITSOutputFile ()
{
  // close the FITS file
  int status = 0;
  fits_close_file(fptr, &status);

  // delete various buffers
  delete dat_wts;
  delete dat_scl;
  delete dat_offs;
  delete dat_freq;
}

//! Get the extension to be added to the end of new filenames
std::string dsp::FITSOutputFile::get_extension () const
{
  return ".sf";
}

void dsp::FITSOutputFile::write_header ()
{

  // a dummy archive to handle all of the extensions
  Reference::To<Pulsar::Archive> archive = new Pulsar::FITSArchive;

  const unsigned nbin  = 1;
  const unsigned nsub  = 1;
  npol = get_input() -> get_npol();
  nchan = get_input() -> get_nchan();

  if (nbit == 0)
    throw Error (InvalidState, "dsp::FITSOutputFile::write_header",
        "nbit was not set");
  nbblk = (nsblk * npol * nchan * nbit)/8;
  cerr << "nbblk="<<nbblk<<endl;
  tblk = double(nsblk) / input -> get_rate();

  if (verbose)
    cerr << "dsp::FITSOutputFile::write_header" << endl
         << "nchan="<<nchan<<" npol="<<npol<<" nsblk="<<nsblk<<" tblk="<<tblk<<" rate="<<input->get_rate()<<" nbit="<<nbit<< endl;

  archive-> resize (nsub, npol, nchan, nbin);

  Pulsar::FITSHdrExtension* ext;
  ext = archive->get<Pulsar::FITSHdrExtension>();
  
  if (ext)
  {
    if (verbose)
      cerr << "dsp::FITSOutputFile::write_header Pulsar::Archive FITSHdrExtension" << endl;

    // Make sure the start time is aligned with pulse phase zero
    // as this is what the PSRFITS format expects.

    MJD initial = get_input()->get_start_time();
    ext->set_start_time (initial);
    ext->set_coordmode("J2000");

    // Set the ASCII date stamp from the system clock (in UTC)

    time_t thetime;
    time(&thetime);
    string time_str = asctime(gmtime(&thetime));

    // Cut off the line feed character
    time_str = time_str.substr(0,time_str.length() - 1);
    ext->set_date_str(time_str);
  }

  archive-> set_telescope ( get_input()->get_telescope() );
  archive-> set_type ( get_input()->get_type() );

  switch (get_input()->get_state())
  {
  case Signal::NthPower:
  case Signal::PP_State:
  case Signal::QQ_State:
    archive->set_state (Signal::Intensity);
    break;
  
  case Signal::FourthMoment:
    archive->set_state (Signal::Stokes);
    break;

  default:
    archive-> set_state ( get_input()->get_state() );
  }

  // probably not correct for search mode
  //archive-> set_scale ( Signal::FluxDensity );

  if (verbose)
    cerr << "dsp::Archiver::set Archive source=" << get_input()->get_source()
         << "\n  coord=" << get_input()->get_coordinates()
         << "\n  bw=" << get_input()->get_bandwidth()
         << "\n  freq=" << get_input()->get_centre_frequency () << endl;

  archive-> set_source ( get_input()->get_source() );
  archive-> set_coordinates ( get_input()->get_coordinates() );
  archive-> set_bandwidth ( get_input()->get_bandwidth() );
  archive-> set_centre_frequency ( get_input()->get_centre_frequency() );
  archive-> set_dispersion_measure ( get_input()->get_dispersion_measure() );

  archive-> set_dedispersed( false );
  archive-> set_faraday_corrected (false);

  // Set any available extensions
  // TODO -- c.f. Archiver -- but I think add the ones we need on an
  // ad hoc basis

  //Pulsar::Backend* backend = archive -> get<Pulsar::Backend>();
  if (verbose)
    cerr << "dsp::FITSOutputFile::write_header; set Pulsar::Backend extension" << endl;
  Pulsar::Backend* backend = new Pulsar::Backend;
  backend->set_name( get_input()->get_machine() );
  // dspsr does not correct Stokes V for lower sideband down conversion
  backend->set_downconversion_corrected( false );
  // dspsr uses the conventional sign for complex phase
  backend->set_argument( Signal::Conventional );
  archive->add_extension( backend );


  // Note, this is now called before the set(Integration,...) call below
  // so that the DigitiserCounts extension gets set up correctly the
  // first time.

  //Pulsar::dspReduction* dspR = archive -> getadd<Pulsar::dspReduction>();
  //if (dspR)
  //{
    //if (verbose > 2)
      //cerr << "dsp::Archiver::set Pulsar::dspReduction extension" << endl;
    //set (dspR);
  //}

  Pulsar::Telescope* telescope = archive -> getadd<Pulsar::Telescope>();
  try
  {
    telescope->set_coordinates ( get_input() -> get_telescope() );
  }
  catch (Error& error)
  {
    if (verbose)
      cerr << "dsp::Archiver WARNING could not set telescope coordinates\n\t"
           << error.get_message() << endl;
  }

  // default Receiver extension
  Pulsar::Receiver* receiver = archive -> getadd<Pulsar::Receiver>();
  receiver->set_name ( get_input() -> get_receiver() );
  receiver->set_basis ( get_input() -> get_basis() );

  //for (unsigned iext=0; iext < extensions.size(); iext++)
  //  archive -> add_extension ( extensions[iext] );

  // set_model must be called after the Integration::MJD has been set

  //archive-> set_filename (get_filename (phase));
  archive -> unload (output_filename);
}

void dsp::FITSOutputFile::write_row ()
{
  if (verbose)
      cerr << "dsp::FITSOutputFile::write_row writing row "<<isub<<endl;
  int status = 0, colnum = 0;
  // write the INDEXVAL
  fits_get_colnum(fptr, CASEINSEN, "INDEXVAL", &colnum, &status);
  fits_write_col(fptr,TINT,colnum,isub,1,1,&isub,&status);
  // write the subint time
  fits_get_colnum(fptr, CASEINSEN, "TSUBINT", &colnum, &status);
  fits_write_col(fptr,TDOUBLE,colnum,isub,1,1, &tblk, &status);
  // write the offset time 
  double offs_sub = tblk/2.0 + isub*tblk;
  fits_get_colnum(fptr, CASEINSEN, "OFFS_SUB", &colnum, &status);
  fits_write_col(fptr,TDOUBLE,colnum,isub,1,1,&offs_sub,&status);
  // write out the data scales, weights, offset
  fits_get_colnum(fptr, CASEINSEN, "DAT_WTS", &colnum, &status);
  fits_write_col(fptr, TFLOAT, colnum, isub, 1, nchan, dat_wts, &status);
  fits_get_colnum(fptr, CASEINSEN, "DAT_SCL", &colnum, &status);
  fits_write_col(fptr, TFLOAT, colnum, isub, 1, nchan*npol, dat_scl, &status);
  cout << status << endl;
  fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &colnum, &status);
  fits_write_col(fptr, TFLOAT, colnum, isub, 1, nchan*npol, dat_offs, &status);
  //fits_write_col(fptr,TDOUBLE,colnum_ra_sub,sub,1,1,&rad,&status);
  //fits_write_col(fptr,TDOUBLE,colnum_dec_sub,sub,1,1,&decd,&status);
  // write the channel frequencies
  fits_get_colnum(fptr, CASEINSEN, "DAT_FREQ", &colnum, &status);
  fits_write_col(fptr,TDOUBLE,colnum,isub,1,nchan,dat_freq,&status);
}

void dsp::FITSOutputFile::initialize ()
{
  if (verbose)
    cerr << "dsp::FITSOutputFile::initialize" << endl;

  if (!dat_wts)
    dat_wts = new float[nchan];
  if (!dat_scl)
    dat_scl = new float[nchan*npol];
  if (!dat_offs)
    dat_offs = new float[nchan*npol];
  if (!dat_freq)
    dat_freq = new double[nchan];
  for (unsigned i; i < nchan; ++i)
  {
    dat_offs[i] = dat_scl[i] = 0;
    dat_wts[i] = 1;
    // TODO -- set dat freqs
    dat_freq[i] = i;
  }
  int status = 0;

  fits_open_file (&fptr,output_filename.c_str(), READWRITE, &status);
  if (status)
    throw Error (FileNotFound, "dsp::FITSOutputFile::initialize",
        "unable to open FITS file for writing");

  fits_movnam_hdu(fptr,BINARY_TBL,"SUBINT",0,&status);
  if (status)
    throw Error (Undefined, "dsp::FITSOutputFIle::initialize",
        "FITS file is missing valid SUBINT table");

  // set up channel-dependent entries with correct size
  int colnum = 0;
  fits_get_colnum(fptr, CASEINSEN, "DAT_FREQ", &colnum, &status);   
  fits_modify_vector_len (fptr, colnum, nchan, &status); 
  fits_get_colnum(fptr, CASEINSEN, "DAT_WTS", &colnum, &status);   
  fits_modify_vector_len (fptr, colnum, nchan, &status); 
  fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &colnum, &status);   
  fits_modify_vector_len (fptr, colnum, nchan*npol, &status); 
  fits_get_colnum(fptr, CASEINSEN, "DAT_SCL", &colnum, &status);   
  fits_modify_vector_len (fptr, colnum, nchan*npol, &status); 

  // set the block (DATA) dim entries

  long naxes[4];
  int naxis=3;
  char tstr[128];
  int ival;

  fits_get_colnum(fptr, CASEINSEN, "DATA", &colnum, &status);  
  fits_modify_vector_len (fptr, colnum, nbblk, &status); 
  naxes[0] = nchan*nbit; 
  naxes[1] = npol;
  naxes[2] = nsblk;

  sprintf(tstr,"TDIM%d",colnum);
  fits_delete_key(fptr, tstr, &status);
  fits_write_tdim(fptr, colnum, naxis, naxes, &status);
  //ival=nsub; fits_update_key(fptr, TINT, "NAXIS2", &ival, NULL, &status );
  fits_update_key(fptr, TINT, "NSBLK", &nsblk, NULL, &status );  
  //fits_report_error(stdout,status);  
}

void dsp::FITSOutputFile::operation ()
{
  if (verbose)
    cerr << "dsp::FITSOutputFile::operation" << endl;

  if (!fptr)
  {
    write_header ();
    initialize ();
  }

  unload_bytes (get_input()->get_rawptr(), get_input()->get_nbytes());

}

int64_t dsp::FITSOutputFile::unload_bytes (const void* void_buffer, uint64_t bytes)
{

  if (verbose)
    cerr << "dsp::FITSOutputFile::unload_bytes" << endl
         << "    bytes="<<bytes<<" nbblk="<<nbblk<<" offset="<<offset<<" isub="<<isub<<endl;
  // cast to char buffer for profit
  unsigned char* buffer = (unsigned char*) void_buffer;

  int colnum = 0, status = 0;
  unsigned to_write = bytes;
  fits_get_colnum (fptr, CASEINSEN, "DATA", &colnum, &status);
  
  // write to incomplete block first
  if (offset)
  {
    if (verbose)
      cerr << "writing to incomplete block" << endl;
    unsigned remainder = nbblk - offset;

    // finish remainder of subint
    if (bytes > remainder)
    {
      fits_write_col_byt (fptr, colnum, isub, offset, remainder, 
          buffer, &status);
      buffer += remainder;
      to_write -= remainder;
      offset = 0;
    }

    // write all available bytes without advancing subint
    else
    {
      fits_write_col_byt (fptr, colnum, isub, offset, bytes, 
          buffer, &status);
      written += bytes;
      offset += bytes;
      return bytes;
    }
  }

  while (to_write >= nbblk)
  {
    if (verbose)
      cerr << "writing to new block" << endl;
    // will give correct one-based index first time through
    isub += 1;

    // write ancillary data for new block/subint
    write_row ();

    // Now write that data into a subintegration in the PSRFITS file
    fits_write_col_byt (fptr, colnum, isub, 1, nbblk, buffer, &status);
    to_write -= nbblk;
    buffer += nbblk;
  }

  // write out remaining bytes to partial subbint
  if (to_write)
  {
    isub += 1;
    write_row();
    fits_write_col_byt (fptr, colnum, isub, 1, to_write, buffer, &status);
    offset += to_write;
  }

  return bytes;

}

void dsp::FITSOutputFile::set_reference_spectrum (Rescale* rescale)
{
  // Reference spectrum packed in PF order
  cerr << "dsp::FITSOutputFile::set_reference_spectrum" << endl;
  for (unsigned ipol = 0; ipol < npol; ++ipol) 
  {
    unsigned offset = ipol * nchan;
    const float* scl = rescale->get_scale (ipol);
    const float* offs = rescale->get_offset (ipol);
    for (unsigned jchan = 0; jchan < nchan; ++jchan)
    {
      dat_scl[jchan+offset] = scl[jchan];
      dat_offs[jchan+offset] = offs[jchan];
    }
  }
}

