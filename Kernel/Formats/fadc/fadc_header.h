/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FADC_HEADER_h
#define __FADC_HEADER_h

/*! fadc_header_get - get header variables

  This is an adjusted version of ascii_header.h for reading 
     FADC headers. The syntax is the same, this version
     just provides a fadc_header_get function for reading 
     the header. 
     In FADC headers, keywords are seperated by colons and 
     values may contain spaces. fadc_header_get works with this 
     format.

  \param header   pointer to the header buffer
  \param keyword  the header keyword, such as NPOL
  \param code     printf/scanf code, such as "%d"

  \retval 0 on success, -1 on failure

  \pre The user must ensure that the code matches the type of the argument.

  For example: EXAMPLE ILLUSTRATES SYNTAX, NOT KEYWORDS !!

  char fadc_header[ASCII_HEADER_SIZE] = ASCII_HEADER_INIT;

  double centre_frequency;
  fadc_header_get (fadc_header, "FREQ", "%lf", &centre_frequency);

  int chan;
  float gain;
  fadc_header_get (fadc_header, "GAIN", "%d %lf", &chan, &centre_frequency);

*/

#ifdef __cplusplus
extern "C" {
#endif

int fadc_header_get (const char* header, const char* keyword,
		      const char* code, ...);

int fadc_blockmap_get (const char* blockmap, const char* keyword, 
		      const char* code, ...);

#ifdef __cplusplus
}
#endif

#endif
