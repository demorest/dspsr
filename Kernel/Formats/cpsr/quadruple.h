/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __quadruple_h
#define __quadruple_h

#ifdef sun
typedef long double quadruple;
#else
typedef struct {double f1; double f2;} quadruple;
#endif

#endif
