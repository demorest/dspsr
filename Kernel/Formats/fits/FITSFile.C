//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <fcntl.h>

#include "Pulsar/Pulsar.h"
#include "Pulsar/Archive.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/Backend.h"
#include "Pulsar/FITSSUBHdrExtension.h"

#include "psrfitsio.h"
#include "fits_params.h"

#include "dsp/FITSFile.h"
#include "dsp/FITSOutputFile.h"
#include "dsp/CloneArchive.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using Pulsar::warning;


dsp::FITSFile::FITSFile (const char* filename)
  : File("FITSFile")
{
  current_byte = 0;
}

bool dsp::FITSFile::is_valid (const char* filename) const
{
  fitsfile* test_fptr = 0;
  int status = 0;

  fits_open_file(&test_fptr, filename, READONLY, &status);
  if (status)
  {
    if (verbose)
    {
      char error[FLEN_ERRMSG];
      fits_get_errstatus(status, error);
      cerr << "FITSFile::is_valid fits_open_file: " << error << endl;
    }
    return false;
  }

  if (verbose)
    cerr << "FITSFile::is_valid test reading MJD" << endl;


  bool result = true;
  try {
    // try to read the MJD from the header
    long day;
    long sec;
    double frac;

    psrfits_read_key (test_fptr, "STT_IMJD", &day);
    psrfits_read_key (test_fptr, "STT_SMJD", &sec);
    psrfits_read_key (test_fptr, "STT_OFFS", &frac);

  }
  catch (Error& error)
  {
    if (verbose)
      cerr << "FITSFile::is_valid failed to read MJD "
        << error.get_message() << endl;
    result = false;
  }

  fits_close_file(test_fptr, &status);
  return result;
}

void read_header(fitsfile* fp, const char* filename, struct fits_params* header)
{
  long day;
  long sec;
  double frac;

  psrfits_read_key(fp, "STT_IMJD", &day);
  psrfits_read_key(fp, "STT_SMJD", &sec);
  psrfits_read_key(fp, "STT_OFFS", &frac);
  header->start_time = MJD((int)day, (int)sec, frac);

  psrfits_move_hdu(fp, "SUBINT");
  psrfits_read_key(fp, "TBIN", &(header->tsamp));
  psrfits_read_key(fp, "NAXIS2", &(header->nrow));
  psrfits_read_key(fp, "NSUBOFFS", &(header->nsuboffs), 0);

  // default is unsigned integers
  psrfits_read_key(fp, "SIGNINT", &(header->signint), 0);
  psrfits_read_key(fp, "ZERO_OFF", &(header->zero_off), 0.0f);
  if (header->zero_off < 0)
    header->zero_off = -header->zero_off;

  /*
  // if unsigned integers used, must have valid zero offset
  if ( (header->signint==0) && (header->zero_off == 0.))
    throw Error(InvalidState, "FITSFile::read_header",
        "Invalid zero offset specified for unsigned data.");
  */

}

void dsp::FITSFile::add_extensions (Extensions* ext)
{
  ext->add_extension (new CloneArchive(archive));
}

void dsp::FITSFile::open_file(const char* filename)
{
  archive = Pulsar::Archive::load(filename);

  unsigned nbits = 0;
  unsigned samples_in_row = 0;
  Reference::To<Pulsar::FITSSUBHdrExtension> ext =
    archive->get<Pulsar::FITSSUBHdrExtension>();

  if (ext) {
    nbits = ext->get_nbits();
    samples_in_row = ext->get_nsblk(); // Samples per row.
  } else {
    throw Error(InvalidState, "FITSFile::open_file",
        "Could not access FITSSUBHdrExtension");
  }

  const unsigned npol  = archive->get_npol();
  const unsigned nchan = archive->get_nchan();

  int status = 0;
  fits_open_file(&fp, filename, READONLY, &status);
  fits_params header;
  read_header(fp, filename, &header);

  signint = header.signint;
  zero_off = header.zero_off;

  get_info()->set_source(archive->get_source());
  get_info()->set_type(Signal::Pulsar);
  get_info()->set_centre_frequency(archive->get_centre_frequency());
  get_info()->set_bandwidth(archive->get_bandwidth());
  get_info()->set_nchan(nchan);
  get_info()->set_npol(npol);

  if (npol == 1 && archive->get_state() != Signal::Intensity)
  {
    warning << "dsp::FITSFile::open_file npol==1 and data state="
            << archive->get_state() << " (reset to Intensity)" << endl;
    get_info()->set_state( Signal::Intensity );
  }
  else
    get_info()->set_state(archive->get_state());

  get_info()->set_nbit(nbits);
  get_info()->set_rate(1.0/header.tsamp);
  get_info()->set_coordinates(archive->get_coordinates());
  get_info()->set_receiver(archive->get<Pulsar::Receiver>()->get_name());
  get_info()->set_basis(archive->get_basis());
  get_info()->set_start_time(header.start_time
      + (uint64_t)header.nsuboffs*(uint64_t)samples_in_row*header.tsamp);
  std::string backend_name = archive->get<Pulsar::Backend>()->get_name();
  if (backend_name == "GUPPI" || backend_name == "PUPPI")
    get_info()->set_machine("GUPPIFITS");
  else if (backend_name == "COBALT")
    get_info()->set_machine("COBALT");
  else
    get_info()->set_machine("FITS");
  get_info()->set_telescope(archive->get_telescope());
  get_info()->set_ndat(header.nrow*samples_in_row);

  set_samples_in_row(samples_in_row);
  set_bytes_per_row((samples_in_row*npol*nchan*nbits) / 8);
  set_number_of_rows(header.nrow);

  data_colnum = dsp::get_colnum(fp, "DATA");
  scl_colnum = dsp::get_colnum(fp, "DAT_SCL");
  offs_colnum = dsp::get_colnum(fp, "DAT_OFFS");

  // Make sure buffer big enough for DAT_SCL/DAT_OFFS
  dat_scl.resize(npol*nchan,1);
  dat_offs.resize(npol*nchan,0);

  fd = ::open(filename, O_RDONLY);
  if (fd < 0) {
    throw Error(FailedSys, "dsp::FITSFile::open",
        "failed open(%s)", filename);
  }
}

int64_t dsp::FITSFile::load_bytes(unsigned char* buffer, uint64_t bytes)
{
  // Column number of the DATA column in the SUBINT table.
  const unsigned nsamp         = get_samples_in_row();

  // Bytes in a row, within the SUBINT table.
  const unsigned bytes_per_row = get_bytes_per_row();

  // Number of rows in the SUBINT table.
  const unsigned nrow          = get_number_of_rows();

  const unsigned nchan = get_info()->get_nchan();
  const unsigned npol  = get_info()->get_npol();
  const unsigned nbit  = get_info()->get_nbit();
  const unsigned bytes_per_sample = (nchan*npol*nbit) / 8;

  // Adjust current_row and byte_offset depending on next sample to read.
  const uint64_t sample = current_byte / bytes_per_sample;

  if (verbose)
    cerr << "dsp::FITSFile::load_bytes load_sample=" << sample
         << " total_samples=" << nsamp * nrow << endl;

  // Calculate the row within the SUBINT table of the target sample to be read.
  unsigned current_row = (int)(sample/nsamp) + 1;

  unsigned char nval = '0';
  int initflag       = 0;
  int status         = 0;

  // TODO: Check for current_row >= && current_row <= nrow

  unsigned byte_offset = (sample % nsamp) * bytes_per_sample;
  unsigned bytes_remaining = bytes;
  unsigned bytes_read = 0;

  while (bytes_remaining > 0 && current_row <= nrow)
  {
    // current_row = [1:nrow]
    if (current_row > nrow)
      throw Error(InvalidState, "FITSFile::load_bytes",
          "current row=%u > nrow=%u", current_row, nrow);

    // Read from byte_offset to end of the row.
    unsigned this_read = bytes_per_row - byte_offset;

    // Ensure we don't read more than expected.
    if (this_read > bytes_remaining) {
      this_read = bytes_remaining;
    }

    if (verbose)
      cerr << "FITSFile::load_bytes row=" << current_row
           << " offset=" << byte_offset << " read=" << this_read << endl;

    // Read the samples
    fits_read_col_byt(fp, data_colnum, current_row, byte_offset+1, 
        this_read, nval, buffer, &initflag, &status);
    if (status) 
    {
      fits_report_error(stderr, status);
      throw FITSError(status, "FITSFile::load_bytes", "fits_read_col_byt");
    }

    // Read the scales
    fits_read_col(fp,TFLOAT,scl_colnum,current_row,1,nchan*npol,
        NULL,&dat_scl[0],NULL,&status);
    if (status) 
    {
      fits_report_error(stderr, status);
      throw FITSError(status, "FITSFile::load_bytes", "fits_read_col");
    }

    // Read the offsets
    fits_read_col(fp,TFLOAT,offs_colnum,current_row,1,nchan*npol,
        NULL,&dat_offs[0],NULL,&status);
    if (status) 
    {
      fits_report_error(stderr, status);
      throw FITSError(status, "FITSFile::load_bytes", "fits_read_col");
    }

    buffer      += this_read;
    byte_offset += this_read;

    // Toggle the 'end of data' flag after the last byte has been read.
    if (current_row == nrow && byte_offset >= bytes_per_row)
      set_eod(true);

    // Adjust byte offset when entire row is read.
    if (byte_offset >= bytes_per_row)
    {
      ++current_row;
      byte_offset = byte_offset % bytes_per_row;
    }

    bytes_remaining -= this_read;
    bytes_read += this_read;
    current_byte += this_read;
  }

  // NB below should technically be inside the above loop, but only the last
  // call has any effect further down the signal path.  We rely on the block
  // size being set such that this method (and update) are called exactly
  // once for each row in the input FITS file.
  update(this);

  return bytes_read;
}

