//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/debug.h

#ifndef __debug_h
#define __debug_h

#ifdef _DEBUG

#include <iostream>
#define DEBUG(x) std::cerr << x << std::endl;

#else

#define DEBUG(x)

#endif

#endif
