/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// This is a bit a of kluge.  Use a psrchive/FITSArchive to handle writing
// out all of the extensions -- this way we can take advantage of the
// development done there.  This all happens in "write header".

// Then we do the data separately, writing rows directly through cfitsio.

// NB -- we need to deal with the reference spectrum, somehow.  It would
// be a REALLY GOOD idea to set tblk to the same value as the Rescale
// operation!

// Recall the PSRFITS output is in TPF order.

// TODO -- since the user can set nbit/nsblk, the instance can be made
// inconsistent if these methods are called after initialize

#ifndef __FITSOutputFile_h
#define __FITSOutputFile_h

#include "dsp/OutputFile.h"
#include <fitsio.h>

namespace dsp {

  class FITSDigitizer;

  int get_colnum (fitsfile* fptr, const char* label);

  //! writes BitSeries data to a PSRFITS "search mode" file
  class FITSOutputFile : public OutputFile 
  {
  public:
   
    //! Construct and open file
    FITSOutputFile (const char* filename=0);

    ~FITSOutputFile ();

    //! Use Rescale callback to set reference spectrum
    void set_reference_spectrum (FITSDigitizer*);

    //! Set the number of samples per output block
    void set_nsblk ( unsigned nblk );

    //! Set the number of bits per output sample
    void set_nbit ( unsigned _nbit );

    //! Set the output filename convention
    void set_atnf ( bool );

    //! Set output mangling
    void set_mangle_output ( bool );

    //! Set length of output file (seconds)
    void set_max_length( double );

  protected:

    //! Need a custom implementation of operation to handle FITS I/O
    virtual void operation ();

    //! Write out all of the FITS extensions.
    void write_header ();

    //! Write metadata after data output and close fits file
    void finalize_fits ();

    //! Get the extension to be added to the end of new filenames
    std::string get_extension () const;

    //! Write nbyte bytes with cfitsio
    virtual int64_t unload_bytes (const void* buffer, uint64_t bytes);

    //! Interface to CFITSIO with error checking and bookkeeping
    unsigned char* write_bytes (int colnum, int isub, int offset, unsigned bytes_to_write, unsigned char** buffer);

    //! samples per block (FITS row)
    unsigned nsblk;

    //! bytes per block
    unsigned nbblk;

    //! time per block
    double tblk;

    //! bits per sample
    unsigned nbit;

    //! convenience store npol
    unsigned npol;

    //! convenience store channel nuumber
    unsigned nchan;

    //! maximum length of output file
    double max_length;

    //! buffer for channels weights
    std::vector<float> dat_wts;

    //! buffer for channel scales
    std::vector<float> dat_scl;

    //! buffer for channel offsets
    std::vector<float> dat_offs;

    //! buffer for channel frequencies
    std::vector<double> dat_freq;

    //! FITS file handle
    fitsfile* fptr;

    //! current subint
    unsigned isub;

  private:

    //! offset into current block to write new data
    unsigned offset;

    //! keep track of bytes written so far
    int64_t written;

    //! optional maximum bytes per file
    int64_t max_bytes;

    //! set up buffers, etc.
    void initialize ();

    //! helper method for FITS output
    void write_row ();

    //! Use ATNF datestr convention
    bool use_atnf;

    //! Use a mangled file name for output; rename on file close
    bool mangle_output;
    std::string mangled_output_filename;

  };

}

#endif 
  
