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
#include "dsp/File.h"

namespace dsp
{
    //! Loads BitSeries data from a FITS data file
    class FITSFile : public File
    {
        public:
            //! Construct and open file
            FITSFile(const char* filename = 0);

            //! Returns true if filename appears to name a valid FITS file
            bool is_valid(const char* filename) const;

        protected:
            //! Open the file
            virtual void open_file(const char* filename);

            virtual int64 load_bytes(unsigned char* buffer, uint64 bytes);

            void set_filename(const std::string fname) {filename = fname;}

            std::string get_filename() {return filename;}

            void set_nsamples(const uint nsamp) {nsamples = nsamp;}

            uint get_nsamples() {return nsamples;}

            std::string filename;
            uint nsamples;

            uint row;

            uint byte_offset;

            void set_bytes_per_row(const uint bytes) {bytes_per_row = bytes;}

            uint get_bytes_per_row() {return bytes_per_row;}

            void set_num_rows(const uint rows) {num_rows = rows;}

            uint get_num_rows() {return num_rows;}

            uint num_rows;

            uint bytes_per_row;

            fitsfile* fp;

            void set_data_colnum(const int colnum) {data_colnum = colnum;}

            int get_data_colnum() {return data_colnum;}

            int data_colnum;

    };
}

float oneBitNumber(int num);
float eightBitNumber(int num);
float fourBitNumber(int num);
float twoBitNumber(int num);

#endif
