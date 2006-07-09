/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CPSR2_HEADER_h
#define __CPSR2_HEADER_h

#define CPSR2_HEADER_SIZE 4096
#define CPSR2_HEADER_VERSION 0.2

/* ************************************************************************

   The CPSR-2 header will be stored in the shared memory of both the
   primary and secondary nodes, as well as in the header of CPSR-2 data
   files written to disk.  It is an ASCII formatted list of fields written
   in "key value" pairs; each pair is separated by one or more newline
   characters.  Comments should be proceeded by '#' characters.  The last 
   string in the header should be the string, "DATA", followed by a single
   newline character.
   
   It is important that keywords (OBSERVATION for example) are not
   replicated in any other part of the file: in the beta version of
   the header reader, we simply examine the header for the first
   occurance of a keyword.  Hiding the replication behind a comment
   symbol is NOT acceptable.

   The header should contain at least the following information:
*/

#define CPSR2_HEADER_INIT \
"CPSR2_HEADER_VERSION 0.2      # Version of this ASCII header\n" \
"CPSR2_DAS_VERSION 0.1         # Version of the Data Acquisition Software\n" \
"CPSR2_FFD_VERSION unset       # Version of the FFD FPGA Software\n" \
"\n" \
"TELESCOPE unset               # telescope name\n" \
"PRIMARY   unset               # primary node host name\n" \
"\n" \
"# time of the rising edge of the first time sample\n" \
"UTC_START unset               # yyyy-mm-dd-hh:mm:ss.fs\n" \
"MJD_START unset               # MJD equivalent to the start UTC\n" \
"\n" \
"OFFSET    unset               # bytes offset from the start MJD/UTC\n" \
"\n" \
"SOURCE    unset               # name of the astronomical source\n" \
"RA        unset               # Right Ascension of the source\n" \
"DEC       unset               # Declination of the source\n" \
"\n" \
"FREQ      unset               # centre frequency on sky in MHz\n" \
"BW        unset               # bandwidth of in MHz (-ve lower sb)\n" \
"TSAMP     unset               # sampling interval in microseconds\n" \
"\n" \
"NBIT      unset               # number of bits per sample\n" \
"NDIM      unset               # dimension of samples (2=complex, 1=real)\n" \
"NPOL      unset               # number of polarizations observed\n" \
"\n"

/*
  
  Binary data begins immediately after the newline character following DATA.

  Programs should initialize a cpsr2 header as follows:

  char cpsr2_header[CPSR2_HEADER_SIZE] = CPSR2_HEADER_INIT;

  You can also:

  strcpy (buffer, CPSR2_HEADER_INIT);

  It is recommended to use the "ascii_header_set/get" routines in
  order to manipulate the CPSR2 header block.  See
  test_cpsr2_header.c, or for example:

  ------------------------------------------------------------------------

  char cpsr2_header[CPSR2_HEADER_SIZE] = CPSR2_HEADER_INIT;

  char* telescope_name = "parkes";
  ascii_header_set (cpsr2_header, "TELESCOPE", "%s", telescope_name);

  float bandwidth = 64.0;
  ascii_header_set (cpsr2_header, "BW", "%f", float);

  [...]

  double centre_frequency;
  ascii_header_get (cpsr2_header, "FREQ", "%lf", &centre_frequency);

*/

#include "ascii_header.h"

#endif
