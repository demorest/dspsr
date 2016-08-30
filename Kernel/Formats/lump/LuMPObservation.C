/***************************************************************************
 *
 *   Copyright (C) 2011, 2013 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LuMPObservation.h"
#include "FilePtr.h"

#include <iostream>
#include <string.h>
#include <cstdlib>

using namespace std;

dsp::LuMPObservation::LuMPObservation (const char* header)
{
  if (header == NULL)
    throw Error (InvalidParam, "LuMPObservation", "no header!");

  load (header);

  // Get the other values
  char buffer[256];

  // ///////////////////////////////////////////////////////////////////
  // Binary Format
  if (ascii_header_get (header, "BINARY_FORMAT", "%s", buffer) < 0) {
      set_binary_format(UnknownBinForm);
  }
  else {
      string s(buffer);
      if(s == "INTEGER_BIN_FORM")
          set_binary_format(IntegerBinForm);
      else if(s == "IEEE_FLOAT")
          set_binary_format(IEEE_FloatBinForm);
      else
          set_binary_format(UnknownBinForm);
  }

  // ///////////////////////////////////////////////////////////////////
  // Endianness
  if (ascii_header_get (header, "ENDIAN", "%s", buffer) < 0) {
      set_data_endianness(UnknownDataEndian);
  }
  else {
      string s(buffer);
      if(s == "LITTLE")
          set_data_endianness(DataLittleEndian);
      else if(s == "BIG")
          set_data_endianness(DataBigEndian);
      else
          set_data_endianness(UnknownDataEndian);
  }

  // ///////////////////////////////////////////////////////////////////
  // Polarization Ordering
  if (ascii_header_get (header, "POL_ORDERING", "%s", buffer) < 0) {
      set_pol_ordering(UnknownOrdering);
  }
  else {
      string s(buffer);
      if(s == "DSPSR")
          set_pol_ordering(DSPSROrdering);
      else if(s == "JONES_MATRIX")
          set_pol_ordering(JonesMatrixOrdering);
      else
          set_pol_ordering(UnknownOrdering);
  }

  // ///////////////////////////////////////////////////////////////////
  // scaling
  double lscale = 1.0; 
  if (ascii_header_get (header, "LUMP_SCALE", "%lf", &lscale) < 0) {
      cerr << "Error: No LUMP_SCALE in LuMP file" << endl;
  }
  set_scale(lscale);

  set_swap( false );

  

  // ///////////////////////////////////////////////////////////////////
  // Start Time
  // Note that DSPSR uses the leading edge of a sample, not the center.
  // The LuMP code puts this into UTC_REFERENCE_TIME_T and UTC_OFFSET_START_LEADING.
  // Note that since LOFAR data do not always start a sample at an exact second boundary,
  // the DSPSR code using the UTC_START information, on exact second boundaries, will
  // fail.
  time_t T0 = 0;
  double T1 = 0.0; // in s
  if (ascii_header_get (header, "UTC_REFERENCE_TIME_T", "%s", buffer) < 0) {
      cerr << "Error: No UTC_REFERENCE_TIME_T in LuMP file" << endl;
  }
  else {
      T0 = time_t(strtoull(buffer,NULL,0));
  }
  if (ascii_header_get (header, "UTC_OFFSET_START_LEADING", "%lf", &T1) < 0) {
      cerr << "Error: No UTC_OFFSET_START_LEADING in LuMP file" << endl;
  }
  set_start_time(MJD(T0) + T1);


  // ///////////////////////////////////////////////////////////////////
  // READ_DATA_FROM_PIPE
  if (ascii_header_get (header, "READ_DATA_FROM_PIPE", "%s", buffer) < 0) {
      set_read_from_LuMP_file(true);
  }
  else {
      if( (buffer[0] == 'T') || (buffer[0] == 't') || (buffer[0] == '1') ) {
          set_read_from_LuMP_file(false);
      }
      else {
          set_read_from_LuMP_file(true);
      }
  }

  if (ascii_header_get (header, "LUMP_VERSION", "%s", buffer) < 0) {
      cerr << "Warning: Unknown LuMP version" << endl;
  }
  else {
      if (verbose)
          cerr << "dsp::LuMPObservation::LuMPObservation LUMP_VERSION=" << buffer << endl;
  }

  if (ascii_header_get (header, "LUMP_MODE", "%s", buffer) < 0) {
      set_mode("Unknown LuMP mode");
      cerr << "Warning: Unknown LuMP mode" << endl;
  }
  else {
      set_mode(buffer);
  }


  // ///////////////////////////////////////////////////////////////////
  // Get the physical number of channels
  unsigned nchan_recorded;
  if (ascii_header_get (header, "NCHAN_RECORDED", "%u", &nchan_recorded) < 0) {
      if (verbose) {
          cerr << "Warning: No NCHAN_RECORDED in LuMP file" << endl;
      }
      nchan_recorded = get_nchan();
  }
  else {
      if((nchan_recorded != get_nchan()) && (get_read_from_LuMP_file())) {
          cerr << "Error: NCHAN_RECORDED(" << nchan_recorded << ") != NCHAN(" << get_nchan() << ") in LuMP file, but READ_DATA_FROM_PIPE is False" << endl;
      }
  }
          


  // ///////////////////////////////////////////////////////////////////
  // Total file size and number of samples
  unsigned hdr_size = 0;
  if (ascii_header_get (header, "HDR_SIZE", "%u", &hdr_size) < 0)
    cerr << "Error: No HDR_SIZE in LuMP file" << endl;
  uint64_t file_size_bytes = 0;
  if (ascii_header_get (header, "FILE_SIZE", "%"PRIu64, &file_size_bytes) < 0) {
      cerr << "Error: No FILE_SIZE in LuMP file" << endl;
  }
  else
  {
    if (verbose)
      cerr << "dsp::LuMPObservation::LuMPObservation " << "file_size_bytes=" << file_size_bytes << endl;
    if(file_size_bytes > 0)
    {
        uint64_t bits = (file_size_bytes - hdr_size) * 8;
        if(nchan_recorded != get_nchan()) {
            uint64_t virtual_bits = bits / nchan_recorded * get_nchan()
                                    + (bits % nchan_recorded) * get_nchan() / nchan_recorded;
            bits = virtual_bits;
            file_size_bytes = bits / 8 + hdr_size;
        }
        uint64_t samples =
            bits
            / uint64_t(get_nbit()*get_npol()*get_nchan()*get_ndim());
        set_ndat(samples);
    }
    else
    {
      set_ndat(0);
    }
  }
  set_LuMP_file_size(file_size_bytes);
  

  set_machine ("LuMP");
}

