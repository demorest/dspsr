//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FITSFile_h
#define __FITSFile_h

#include "fitsio.h"

#include "Pulsar/Archive.h"

#include "dsp/File.h"

namespace dsp
{
  //! Loads BitSeries data from a PSRFITS data file
  class FITSFile : public File
  {
    public:
      //! Construct and open file
      FITSFile(const char* filename = 0);

      //! Returns true if filename appears to name a valid FITS file
      bool is_valid(const char* filename) const;

      void add_extensions (Extensions*);

    protected:
      //! Open the file
      virtual void open_file(const char* filename);

      //! Load nbyte bytes of sampled data from the device into buffer.
      virtual int64_t load_bytes(unsigned char* buffer, uint64_t bytes);

      void set_samples_in_row(const unsigned _samples_in_row) { samples_in_row =
        _samples_in_row; }

      unsigned get_samples_in_row() const { return samples_in_row; }

      void set_bytes_per_row(const unsigned bytes) { bytes_per_row = bytes; }

      unsigned get_bytes_per_row() { return bytes_per_row; }

      void set_data_colnum(const int colnum) { data_colnum = colnum; }

      int get_data_colnum() const { return data_colnum; }

      Reference::To<Pulsar::Archive> archive;

      //! Column number of the DATA column in the SUBINT table.
      int data_colnum;

      //! Store the instance of fitsfile, so it is only opened once.
      fitsfile* fp;

      //! Number of samples per row
      unsigned samples_in_row;

      //! Number of bytes per row in the SUBINT table.
      unsigned bytes_per_row;

      //! The value applied to the data to make sure they produce a
      //  zero-centred mean.
      unsigned zero_offset;
  };
}

#endif
