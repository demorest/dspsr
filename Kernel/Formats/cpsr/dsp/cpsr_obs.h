#ifndef __CPSR_H
#define __CPSR_H

#include <vector>
#include <string>
#include "psr_cpp.h"

#define cpsr 1
#include "pspm_search_header.h"

#define TAPES_LOG "cpsrdata/cpsrtapes.xfer"
#define DLT_LOCK "/tmp/DLT.lock"

class MJD;
class fptm_obs;

// these functions deal with transfering CPSR data to disk and looking
// up observations in whatever databases are available (and fashionable)

int cpsr_lookup (const char* filename, const char* tapename,
		 string* cpsrname, vector<int>* targets);

int cpsr_log_tapename (const char* filename, 
		       const char* tapename, const char* cpsrname);

int cpsr_log_done (const char* filename, const char* tapename,
		   const vector<int>& targets);

int cpsr_log_tapefin (const char* filename, const char* tapename, int fend);

int cpsr_log_tape_error (const char* log_filename, const char* tapename);

int cpsr_log_scandone (const char* filename);

int cpsr_log_scanning (const char* filename);

int file_relevant (vector<fptm_obs>* observations, const char* filename,
		   string* identifier=NULL);
int tape_relevant (vector<fptm_obs>* observations,
		   const char* tapedev, int filenum,
		   string* identifier=NULL);
int fd_relevant   (vector<fptm_obs>* observations, int fd,
		   string* identifier=NULL);
int PSPM_relevant (vector<fptm_obs>* observations, PSPM_SEARCH_HEADER* header,
		   string* identifier=NULL);

#endif // __CPSR_H
