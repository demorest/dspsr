/***************************************************************************
 *
 *   Copyright (C) 2000 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
// dspsr/Kernel/Formats/cpsr/pspm++.h

#ifndef __PSPM_H
#define __PSPM_H

#include <string>

#include "MJD.h"

#define cpsr 1
#include "pspm_search_header.h"

/*
 * CPSR Sideband modes copied from pspm.h
 */
#define UNKNOWN_SIDEBAND        0
#define SSB_LOWER               1
#define SSB_UPPER               2
#define DSB_SKYFREQ             3
#define DSB_REVERSED            4

// these functions return some parameters that must be derived from the
// fields in a PSPM_SEARCH_HEADER struct.  Most others are trivially
// obtained directly from the fields

MJD    PSPMstart_time (const PSPM_SEARCH_HEADER* header);
double PSPMduration (const PSPM_SEARCH_HEADER* header);

// this function will perform some sanity checks and "correct"
// (ie. set to default) certain values if inconsistencies are found.
bool   PSPMverify (PSPM_SEARCH_HEADER* hdr, bool verbose = false);

std::string PSPMidentifier (const PSPM_SEARCH_HEADER* hdr);
std::string PSPMsource (const PSPM_SEARCH_HEADER* hdr);

// these return a pointer to a privately managed location.
// do not try to delete
PSPM_SEARCH_HEADER* pspm_read (const char* filename);
PSPM_SEARCH_HEADER* pspm_read (const char* tapedev, int filenum);
PSPM_SEARCH_HEADER* pspm_read (int fd);

#endif
