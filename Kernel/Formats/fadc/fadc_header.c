/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "fadc_header.h"

#define STRLEN 128

// static char* whitespace = " \t\n";

int fadc_header_get (const char* header, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char* value = NULL;
  int ret = 0;

  /* find the keyword */
  char* key = strstr (header, keyword);
  if (!key)
    return -1;

  /* find the colon after the keyword and go 1 character further*/
  //  value = key + strcspn (key, whitespace);
  value = key + strcspn (key, ":") +1;
  
  /* parse the value */
  va_start (arguments, format);
  ret = vsscanf (value, format, arguments);
  va_end (arguments);

  return ret;
}


int fadc_blockmap_get (const char* blockmap, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char* value = 0;
  int ret = 0;

  /* find the keyword */
  char* key = strstr (blockmap, keyword);
  if (!key)
    return -1;

  /* find the character after the keyword */
  value = key + strlen (keyword);

  /* parse the value */
  va_start (arguments, format);
  ret = vsscanf (value, format, arguments);
  va_end (arguments);

  return ret;
}


