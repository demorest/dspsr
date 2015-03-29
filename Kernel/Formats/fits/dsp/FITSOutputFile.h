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

#ifndef __FITSOutputFile_h
#define __FITSOutputFile_h

#include "dsp/OutputFile.h"
#include <fitsio.h>

namespace dsp {

  class Rescale;

  //! writes BitSeries data to a PSRFITS "search mode" file
  class FITSOutputFile : public OutputFile 
  {
  public:
   
    //! Construct and open file
    FITSOutputFile (unsigned nbit, const char* filename=0);

    ~FITSOutputFile ();

    void set_reference_spectrum (Rescale*);

  protected:

    //! Need a custom implementation of operation to handle FITS I/O
    virtual void operation ();

    //! Write out all of the FITS extensions.
    void write_header ();

    //! Get the extension to be added to the end of new filenames
    std::string get_extension () const;

    //! Write nbyte bytes with cfitsio
    virtual int64_t unload_bytes (const void* buffer, uint64_t bytes);

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

    //! buffer for channels weights
    float* dat_wts;

    //! buffer for channel offsets
    float* dat_offs;

    //! buffer for channel scales
    float* dat_scl;

    //! buffer for channel frequencies
    double* dat_freq;

    //! FITS file handle
    fitsfile* fptr;

    //! current subint
    unsigned isub;

  private:

    //! offset into current block to write new data
    unsigned offset;

    //! keep track of bytes written so far
    uint64_t written;

    //! set up buffers, etc.
    void initialize ();

    //! helper method for FITS output
    void write_row ();



  };

}

#endif 
  
