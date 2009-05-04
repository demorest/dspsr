/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FITSFile.h"
#include "dsp/CloneArchive.h"

#include "psrfitsio.h"
#include "fits_params.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/FITSSUBHdrExtension.h"

#include <fcntl.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

dsp::FITSFile::FITSFile (const char* filename)
  : File("FITSFile"), 
    current_row(1),
    byte_offset(0)
{
}

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

    } catch (Error& error) {
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

    int status = 0;
    fits_movnam_hdu(fp, BINARY_TBL, "SUBINT", 0, &status);
    psrfits_read_key(fp, "TBIN", &(header->tsamp));

    if (status)
        throw FITSError(status, "FITSFile - read_header",
                "fits_read_key (TBIN)");

    psrfits_read_key(fp, "NAXIS2", &(header->nsubint));

    if (status)
        throw FITSError(status, "FITSFile - read_header",
                "fits_read_key (NAXIS2)");
}

void dsp::FITSFile::add_extensions (Extensions* ext)
{
  ext->add_extension (new CloneArchive(archive));
}

void dsp::FITSFile::open_file(const char* filename)
{
    archive = Pulsar::Archive::load(filename);

    Reference::To<Pulsar::FITSSUBHdrExtension> ext =
        archive->get<Pulsar::FITSSUBHdrExtension>();

    const uint nbits = ext->get_nbits();
    const uint nsamp = ext->get_nsblk();
    const uint npol = archive->get_npol();
    const uint nchan = archive->get_nchan();

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

    switch (npol) {
        case 1:
            info.set_state (Signal::Intensity);
            break;
        case 2:
            info.set_state (Signal::PPQQ);
            break;
        case 4:
            info.set_state (Signal::Coherence);
            info.set_ndim(1);
            info.set_npol(4);
            break;
        default:
            throw Error(InvalidState, "FITSFile::open_file",
                    "invalid header npol=%d", npol);
    }

    info.set_rate(1.0 / header.tsamp);
    info.set_coordinates( archive->get_coordinates() );
    info.set_receiver( archive->get<Pulsar::Receiver>()->get_name() );
    info.set_basis( archive->get_basis() );
    info.set_start_time(header.start_time);
    info.set_machine("FITS");
    info.set_telescope(archive->get_telescope());
    info.set_ndat(header.nsubint * nsamp);

    set_nsamples(nsamp);
    set_bytes_per_row( (nsamp * npol * nchan * nbits) / 8 );

    int colnum;
    fits_get_colnum(fp, CASEINSEN, "DATA", &colnum, &status);

    if (status)
        throw FITSError(status, "FITSFile::open_file",
                "fits_get_colnum (DATA)");

    set_data_colnum(colnum);

    fd = ::open(filename, O_RDONLY);
    if (fd < 0)
        throw Error(FailedSys, "dsp::FITSFile::open",
                "failed open(%s)", filename);
}

int64 dsp::FITSFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
    const int colnum = get_data_colnum();
    const uint nsamp = get_nsamples();
    const uint nsub = info.get_ndat() / nsamp;

    // bytes to read each row 
    const uint bytes_per_row = get_bytes_per_row();

	    uint bytes_to_read = bytes;

	    while (bytes_to_read)
	    {
	      if (current_row > nsub)
		throw Error (InvalidState, "dsp::FITSFile::load_bytes",
			     "current row=%u > nrow=%u", current_row, nsub);

      // read up to the end of the current row or the number of bytes to read
      uint this_read = bytes_per_row - byte_offset;
      if (this_read > bytes_to_read)
	this_read = bytes_to_read;

      // read from the current row offset by byte_offset
      unsigned char nval = '0';
      int initflag = 0;
      int status = 0;

      if (verbose)
        cerr << "dsp::FITSFile::load_bytes row=" << current_row 
	     << " offset=" << byte_offset << " read=" << this_read << endl;

      fits_read_col_byt (fp, colnum, current_row, byte_offset+1, this_read, nval,
			 buffer, &initflag, &status);

      if (status)
	throw FITSError (status, "dsp::FITSFile::load_bytes",
			 "fits_read_col_byt");

      // offset the base pointer and byte_offset in the current row
      buffer += this_read;
      byte_offset += this_read;

      if (byte_offset == bytes_per_row)
      {
        ++current_row;
	byte_offset = 0;
      }

      bytes_to_read -= this_read;
    }

    return bytes;
}
