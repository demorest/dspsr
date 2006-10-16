/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ASCII_HEADER_h
#define __ASCII_HEADER_h

/*! ascii_header_set/get - Set/get header variables

  \param header   pointer to the header buffer
  \param keyword  the header keyword, such as NPOL
  \param code     printf/scanf code, such as "%d"

  \retval 0 on success, -1 on failure

  \pre The user must ensure that the code matches the type of the argument.

  For example:

  char ascii_header[ASCII_HEADER_SIZE] = ASCII_HEADER_INIT;

  char* telescope_name = "parkes";
  ascii_header_set (ascii_header, "TELESCOPE", "%s", telescope_name);

  float bandwidth = 64.0;
  ascii_header_set (ascii_header, "BW", "%f", float);

  [...]

  double centre_frequency;
  ascii_header_get (ascii_header, "FREQ", "%lf", &centre_frequency);

  int chan;
  float gain;
  ascii_header_get (ascii_header, "GAIN", "%d %lf", &chan, &centre_frequency);

*/

#ifdef __cplusplus
extern "C" {
#endif

int ascii_header_set (char* header, const char* keyword,
		      const char* code, ...);

int ascii_header_get (const char* header, const char* keyword,
		      const char* code, ...);

#ifdef __cplusplus
}
#endif

#endif
