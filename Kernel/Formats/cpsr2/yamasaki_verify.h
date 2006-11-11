/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __YAMASAKI_VERIFY_h
#define __YAMASAKI_VERIFY_h

#include "environ.h"
int yamasaki_verify (const char* filename, uint64 offset_bytes,
		     uint64 search_offset);

#endif
