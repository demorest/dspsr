/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "ascii_header.h"
#include "dada_def.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define STRLEN 4096

static char* whitespace = " \t\n";

// search header for keyword and ensure that it is preceded by whitespace */
char* ascii_header_find (const char* header, const char* keyword)
{
  char* key = strstr (header, keyword);

  // keyword might be the very first word in header
  while (key > header)
  {
    // fprintf (stderr, "found=%s", key);

    // if preceded by a new line, return the found key
    if ( ((*(key-1) == '\n') || (*(key-1) == '\\')) && 
         ((*(key+strlen(keyword)) == '\t') || (*(key+strlen(keyword)) == ' ')))
      break;

    // otherwise, search again, starting one byte later
    key = strstr (key+1, keyword);
  }

  return key;
}

int ascii_header_set (char* header, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char value[STRLEN];
  char* eol = 0;
  char* dup = 0;
  int ret = 0;

  /* find the keyword (also the insertion point) */
  char* key = ascii_header_find (header, keyword);  

  if (key) {
    /* if the keyword is present, find the first '#' or '\n' to follow it */
    eol = key + strcspn (key, "#\n");
  }
  else {
    /* if the keyword is not present, append to the end, before "DATA" */
    eol = strstr (header, "DATA\n");
    if (eol)
      /* insert in front of DATA */
      key = eol;
    else
      /* insert at end of string */
      key = header + strlen (header);
  }

  va_start (arguments, format);
  ret = vsnprintf (value, STRLEN, format, arguments);
  va_end (arguments);

  if (ret < 0) {
    perror ("ascii_header_set: error snprintf\n");
    return -1;
  }

  if (eol)
    /* make a copy */
    dup = strdup (eol);

  /* %Xs dictates only a minumum string length */
  if (sprintf (key, "%-12s %-20s   ", keyword, value) < 0) {
    if (dup)
      free (dup);
    perror ("ascii_header_set: error sprintf\n");
    return -1;
  }

  if (dup) {
    strcat (key, dup);
    free (dup);
  }
  else
    strcat (key, "\n");

  return 0;
}

int ascii_header_get (const char* header, const char* keyword,
		      const char* format, ...)
{
  va_list arguments;

  char* value = 0;
  int ret = 0;

  /* find the keyword */
  char* key = ascii_header_find (header, keyword);
  if (!key)
    return -1;

  /* find the value after the keyword */
  value = key + strcspn (key, whitespace);

  /* parse the value */
  va_start (arguments, format);
  ret = vsscanf (value, format, arguments);
  va_end (arguments);

  return ret;
}

int ascii_header_del (char * header, const char * keyword)
{
  /* find the keyword (also the delete from point) */
  char * key = ascii_header_find (header, keyword);

  /* if the keyword is present, find the first '#' or '\n' to follow it */
  if (key) 
  {
    char * eol = key + strcspn (key, "\n") + 1;

    // make a copy of everything after the end of the key we are deleting
    char * dup = strdup (eol);

    if (dup) 
    {
      key[0] = '\0';
      strcat (header, dup);
      free (dup);
      return 0;
    }
    else
      return -1;
  }
  else
    return -1;
}

size_t ascii_header_get_size (char * filename)
{
  size_t hdr_size = -1;
  int fd = open (filename, O_RDONLY);
  if (!fd)
  {
    fprintf (stderr, "ascii_header_get_size: failed to open %s for reading\n", filename);
  }
  else
  {
    hdr_size = ascii_header_get_size_fd (fd);
    close (fd);
  }
  return hdr_size;
}

size_t ascii_header_get_size_fd (int fd)
{
  size_t hdr_size = -1;
  char * header = (char *) malloc (DADA_DEFAULT_HEADER_SIZE+1);
  if (!header)
  {
    fprintf (stderr, "ascii_header_get_size: failed to allocate %d bytes\n", DADA_DEFAULT_HEADER_SIZE+1);
  }
  else
  {
    // seek to start of file
    lseek (fd, 0, SEEK_SET);

    // read the header 
    ssize_t ret = read (fd, header, DADA_DEFAULT_HEADER_SIZE);
    if (ret != DADA_DEFAULT_HEADER_SIZE)
    {
      fprintf (stderr, "ascii_header_get_size: failed to read %d bytes from file\n", DADA_DEFAULT_HEADER_SIZE);
    }
    else
    {
      // check the actual HDR_SIZE in the header
      if (ascii_header_get (header, "HDR_SIZE", "%ld", &hdr_size) != 1)
      {
        fprintf (stderr, "ascii_header_get_size: failed to read HDR_SIZE from header\n");
        hdr_size = -1;
      }
    }
    // seek back to start of file
    lseek (fd, 0, SEEK_SET);
    free (header);
  }
  return hdr_size;
}

