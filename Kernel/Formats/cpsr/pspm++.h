/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/pspm++.h,v $
   $Revision: 1.1 $
   $Date: 2000/01/13 03:56:26 $
   $Author: wvanstra $ */

#ifndef __PSPM_H
#define __PSPM_H

#include <string>
#include "MJD.h"

// these function return some parameter that must be derived from the
// fields in a PSPM_SEARCH_HEADER struct.  Most others are trivially
// obtained directly from the fields

MJD    PSPMstart_time (const PSPM_SEARCH_HEADER* header);
double PSPMduration (const PSPM_SEARCH_HEADER* header);
bool   PSPMverify (const PSPM_SEARCH_HEADER* hdr);
string PSPMidentifier (const PSPM_SEARCH_HEADER* hdr);
string PSPMsource (const PSPM_SEARCH_HEADER* hdr);

// these return a pointer to a privately managed location.
// do not try to delete
PSPM_SEARCH_HEADER* pspm_read (const char* filename);
PSPM_SEARCH_HEADER* pspm_read (const char* tapedev, int filenum);
PSPM_SEARCH_HEADER* pspm_read (int fd);

#endif
