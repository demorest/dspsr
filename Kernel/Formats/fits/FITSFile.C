//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <fcntl.h>

#include "Pulsar/Archive.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/FITSSUBHdrExtension.h"

#include "psrfitsio.h"
#include "fits_params.h"

#include "dsp/FITSFile.h"
#include "dsp/CloneArchive.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

dsp::FITSFile::FITSFile (const char* filename)
  : File("FITSFile")
{}

bool dsp::FITSFile::is_valid (const char* filename) const
{
  fitsfile* test_fptr = 0;
  int status = 0;

  fits_open_file(&test_fptr, filename, READONLY, &status);
  if (status) {
    if (verbose) {
      char error[FLEN_ERRMSG];
      fits_get_errstatus(status, error);
      cerr << "FITSFile::is_valid fits_open_file: " << error << endl;
    }
    return false;
  }

  if (verbose) {
    cerr << "FITSFile::is_valid test reading MJD" << endl;
  }

  bool result = true;
  try {
    // try to read the MJD from the header
    long day;
    long sec;
    double frac;

    psrfits_read_key (test_fptr, "STT_IMJD", &day);
    psrfits_read_key (test_fptr, "STT_SMJD", &sec);
    psrfits_read_key (test_fptr, "STT_OFFS", &frac);

  } catch (Error& error) {
    if (verbose) {
      cerr << "FITSFile::is_valid failed to read MJD "
        << error.get_message() << endl;
    }
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

  int status = 0;
  fits_movnam_hdu(fp, BINARY_TBL, "SUBINT", 0, &status);
  psrfits_read_key(fp, "TBIN", &(header->tsamp));

  if (status) {
    throw FITSError(status, "FITSFile - read_header",
        "fits_read_key (TBIN)");
  }

  psrfits_read_key(fp, "NAXIS2", &(header->nrow));

  if (status) {
    throw FITSError(status, "FITSFile - read_header",
        "fits_read_key (NAXIS2)");
  }
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

  info.set_source(archive->get_source());
  info.set_type(Signal::Pulsar);
  info.set_centre_frequency(archive->get_centre_frequency());
  info.set_bandwidth(archive->get_bandwidth());
  info.set_nchan(nchan);
  info.set_nbit(nbits);
  info.set_state(archive->get_state());
  info.set_rate(1.0/header.tsamp);
  info.set_coordinates(archive->get_coordinates());
  info.set_receiver(archive->get<Pulsar::Receiver>()->get_name());
  info.set_basis(archive->get_basis());
  info.set_start_time(header.start_time);
  info.set_machine("FITS");
  info.set_telescope(archive->get_telescope());
  info.set_ndat(header.nrow*samples_in_row);

  set_samples_in_row(samples_in_row);
  set_bytes_per_row((samples_in_row*npol*nchan*nbits) / 8);

  int colnum;
  fits_get_colnum(fp, CASEINSEN, "DATA", &colnum, &status);

  if (status) {
    throw FITSError(status, "FITSFile::open_file",
        "fits_get_colnum (DATA)");
  }

  set_data_colnum(colnum);

  fd = ::open(filename, O_RDONLY);
  if (fd < 0) {
    throw Error(FailedSys, "dsp::FITSFile::open",
        "failed open(%s)", filename);
  }
}

int64_t dsp::FITSFile::load_bytes(unsigned char* buffer, uint64_t bytes)
{
  // Column number of the DATA column in the SUBINT table.
  const int colnum             = get_data_colnum();
  const unsigned nsamp         = get_samples_in_row();

  // Bytes in a row, within the SUBINT table.
  const unsigned bytes_per_row = get_bytes_per_row();

  // Number of rows in the SUBINT table.
  const unsigned nrow          = info.get_ndat()/nsamp;

  // Adjust current_row and byte_offset depending on next sample to read.
  const uint64_t sample = get_load_sample();

  // Calculate the row within the SUBINT table of the target sample to be read.
  unsigned current_row = (int)(sample/(nsamp)) + 1;

  const unsigned nchan = info.get_nchan();
  const unsigned npol  = info.get_npol();
  const unsigned nbit  = info.get_nbit();
  const unsigned bytes_per_sample = (nchan*npol*nbit) / 8;

  unsigned char nval = '0';
  int initflag       = 0;
  int status         = 0;

  // TODO: Check for current_row >= && current_row <= nrow

  unsigned byte_offset = (sample % nsamp) * bytes_per_sample;
  unsigned bytes_remaining = bytes;

  while (bytes_remaining > 0) {
    // current_row = [1:nrow]
    if (current_row > nrow) {
      throw Error(InvalidState, "FITSFile::load_bytes",
          "current row=%u > nrow=%u", current_row, nrow);
    }

    // Read from byte_offset to end of the row.
    unsigned this_read = bytes_per_row - byte_offset;

    // Ensure we don't read more than expected.
    if (this_read > bytes_remaining) {
      this_read = bytes_remaining;
    }

    if (verbose) {
      cerr << "FITSFile::load_bytes row=" << current_row
        << " offset=" << byte_offset << " read=" << this_read << endl;
    }

    fits_read_col_byt(fp, colnum, current_row, byte_offset+1, this_read, nval,
        buffer, &initflag, &status);

    if (status) {
      fits_report_error(stderr, status);
      throw FITSError(status, "FITSFile::load_bytes", "fits_read_col_byt");
    }

    buffer      += this_read;
    byte_offset += this_read;

    // Toggle the 'end of data' flag after the last byte has been read.
    if (current_row == nrow & byte_offset >= bytes_per_row) {
      set_eod(true);
    }

    // Adjust byte offset when entire row is read.
    if (byte_offset >= bytes_per_row) {
      ++current_row;
      byte_offset = byte_offset % bytes_per_row;
    }

    bytes_remaining -= this_read;
  }

  return bytes;
}
