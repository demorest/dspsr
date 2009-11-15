//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/debug.h,v $
   $Revision: 1.1 $
   $Date: 2009/11/15 00:47:21 $
   $Author: straten $ */

#ifndef __debug_h
#define __debug_h

#ifdef _DEBUG

#include <iostream>
#define DEBUG(x) std::cerr << x << std::endl;

#else

#define DEBUG(x)

#endif

#endif
