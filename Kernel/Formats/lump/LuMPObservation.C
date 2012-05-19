/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
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
  if (ascii_header_get (header, "UTC_OFFSET_START_LEADING", "%s", &T1) < 0) {
      cerr << "Error: No UTC_OFFSET_START_LEADING in LuMP file" << endl;
  }
  set_start_time(MJD(T0) + T1);




  if (ascii_header_get (header, "LUMP_MODE", "%s", buffer) < 0) {
      set_mode("Unknown LuMP mode");
      cerr << "Warning: Unknown LuMP mode" << endl;
  }
  else {
      set_mode(buffer);
  }
  set_machine ("LuMP");
}

