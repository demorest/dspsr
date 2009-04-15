/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FITSFile.h"
#include "dsp/psrfitsio.h"
#include "fits_params.h"

#include <fcntl.h>

#include "Pulsar/Archive.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/FITSSUBHdrExtension.h"

#include "dsp/CloneArchive.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

dsp::FITSFile::FITSFile(const char* filename) : File("FITSFile"), row(1),
    byte_offset(0) {}

bool dsp::FITSFile::is_valid(const char* filename) const
{
    fitsfile* test_fptr = 0;
    int status = 0;
    char error[FLEN_ERRMSG];

    fits_open_file(&test_fptr, filename, READONLY, &status);
    if (status) {
        if (verbose) {
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
    psrfits_read_key(fp, "STT_IMJD", &(header->day));
    psrfits_read_key(fp, "STT_SMJD", &(header->sec));
    psrfits_read_key(fp, "STT_OFFS", &(header->frac));

    int status = 0;
    fits_movnam_hdu(fp, BINARY_TBL, "SUBINT", 0, &status); 

    psrfits_read_key(fp, "TBIN", &(header->tsamp));
    psrfits_read_key(fp, "NAXIS2", &(header->nsubint));
}

void dsp::FITSFile::open_file(const char* filename)
{
    Reference::To<Pulsar::Archive> archive = Pulsar::Archive::load(filename);
    info.add_extension(new CloneArchive(archive));

    Reference::To<Pulsar::FITSSUBHdrExtension> ext =
        archive->get<Pulsar::FITSSUBHdrExtension>();

    const uint nbits = ext->get_nbits();
    const uint nsamp = ext->get_nsblk();
    const uint npol = archive->get_npol();
    const uint nchan = archive->get_nchan();

    set_filename(filename);

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

    MJD startTime = MJD((int)header.day, (int)header.sec, header.frac);
    info.set_start_time(startTime);

    info.set_machine("FITS");
    info.set_telescope(archive->get_telescope());

    info.set_ndat(header.nsubint * nsamp);
    set_nsamples(nsamp);

    set_bytes_per_row((nsamp * npol *
                nchan) / (8 / nbits));

    fd = ::open(filename, O_RDONLY);
    if (fd < 0)
        throw Error(FailedSys, "dsp::FITSFile::open",
                "failed open(%s)", filename);

    int colnum;
    fits_get_colnum(fp, CASEINSEN, "DATA", &colnum, &status);
    set_data_colnum(colnum);
}

int64 dsp::FITSFile::load_bytes(unsigned char* buffer, uint64 bytes)
{
    cerr << "dsp::FITSFile::load_bytes bytes: " << bytes << endl;

    const int colnum = get_data_colnum();
    int status = 0;

    const uint nsamp = get_nsamples();
    const uint nsub = info.get_ndat() / nsamp;

    unsigned char nval = '0';
    int initflag = 0;

    // bytes to read each subint
    const uint bytes_per_subint = get_bytes_per_row();
    cerr << "bytes_per_subint: " << bytes_per_subint << endl;

    uint bytes_to_read = bytes;
    uint i = 0;

    if (byte_offset) {
        // 1. read the remainder of the row
        const uint remainder = bytes_per_subint - byte_offset;

        fits_read_col_byt(fp, colnum, row, byte_offset, remainder, nval,
                &buffer[i], &initflag, &status);

        i += remainder;
        ++row;
        bytes_to_read -= remainder;
    }

    // 2. read rows while it's possible to read complete rows
    while (bytes_to_read > bytes_per_subint) {
        fits_read_col_byt(fp, colnum, row, 1, bytes_per_subint, nval,
                &buffer[i], &initflag, &status);

        i += bytes_per_subint;
        ++row;
        bytes_to_read -= bytes_per_subint;
    }

    byte_offset = bytes_to_read;

    // ensure we don't read past the end of data table
    const uint total_bytes = nsub * bytes_per_subint;
    uint read_final_bytes;
    if (row * bytes_per_subint + byte_offset > total_bytes)
        read_final_bytes = total_bytes - row * bytes_per_subint;
    else
        read_final_bytes = byte_offset;

    // 3.
    fits_read_col_byt(fp, colnum, row, 1, read_final_bytes, nval,
            &buffer[i], &initflag, &status);

    printf("buffer[0]: %d\n", buffer[0]);
    printf("buffer[1]: %d\n", buffer[1]);
    printf("buffer[bytes]: %d\n", buffer[bytes]);
    printf("buffer[bytes - 1]: %d\n\n", buffer[bytes - 1]);

    // old code
    /*for (uint isub = 1; isub <= nsub; ++isub) {
        fits_read_col_byt(fp, colnum, isub, 1, bytes_per_subint, nval,
                &buffer[i], &initflag, &status);

        i += bytes_per_subint;
    }*/

    return i + read_final_bytes;
}
