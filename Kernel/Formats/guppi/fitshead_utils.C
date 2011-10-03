//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// Simple overloaded C++ interface to the hget routines

#include "fitshead.h"
#include "fitshead_utils.h"

int hget (const char *buffer, const char *key, int *val)
{
  return hgeti4 (buffer, key, val);
}

int hget (const char *buffer, const char *key, long long *val)
{
  // This one is a bit of a hack since there is no hgeti8...
  double tmp;
  int rv = hgetr8 (buffer, key, &tmp);
  *val = (long long)tmp;
  return rv;
}

int hget (const char *buffer, const char *key, float *val)
{
  return hgetr4 (buffer, key, val);
}

int hget (const char *buffer, const char *key, double *val)
{
  return hgetr8 (buffer, key, val);
}

int hget (const char *buffer, const char *key, char *val)
{
  // Also a bit hacky.. assumes 80 chars have been reserved for val
  // could do this better with string.
  return hgets (buffer, key, 80, val);
}
