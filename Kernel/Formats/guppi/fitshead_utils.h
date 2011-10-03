//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef _fitshead_utils_h
#define _fitshead_utils_h

// Simple overloaded C++ interface to the hget routines

#include "fitshead.h"

int hget (const char *buffer, const char *key, int *val);
int hget (const char *buffer, const char *key, long long *val);
int hget (const char *buffer, const char *key, float *val);
int hget (const char *buffer, const char *key, double *val);
int hget (const char *buffer, const char *key, char *val);

#endif
