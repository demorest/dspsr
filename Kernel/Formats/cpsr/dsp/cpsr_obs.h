/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __CPSR_H
#define __CPSR_H

#include <vector>
#include <string>

#define cpsr 1
#include "pspm_search_header.h"

#define TAPES_LOG "cpsrdata/cpsrtapes.xfer"
#define DLT_LOCK "/tmp/DLT.lock"

class MJD;
class fptm_obs;

// these functions deal with transfering CPSR data to disk and looking
// up observations in whatever databases are available (and fashionable)

// default file log name
const char* cpsr_default_log ();

int cpsr_lookup (const char* filename, const char* tapename,
		 std::string* cpsrname, std::vector<int>* targets);

int cpsr_log_tapename (const char* filename, 
		       const char* tapename, const char* cpsrname);

int cpsr_log_done (const char* filename, const char* tapename,
		   const std::vector<int>& targets);

int cpsr_log_tapefin (const char* filename, const char* tapename, int fend);

int cpsr_log_tape_error (const char* log_filename, const char* tapename);

int cpsr_log_scandone (const char* filename);

int cpsr_log_scanning (const char* filename);

int file_relevant (std::vector<fptm_obs>* observations, const char* filename,
		   std::string* identifier=NULL);
int tape_relevant (std::vector<fptm_obs>* observations,
		   const char* tapedev, int filenum,
		   std::string* identifier=NULL);
int fd_relevant   (std::vector<fptm_obs>* observations, int fd,
		   std::string* identifier=NULL);
int PSPM_relevant (std::vector<fptm_obs>* observations,
		   PSPM_SEARCH_HEADER* header,
		   std::string* identifier=NULL);

#endif // __CPSR_H
