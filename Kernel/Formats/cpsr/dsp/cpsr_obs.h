#ifndef __CPSR_H
#define __CPSR_H

#include <vector>
#include <string>

#define cpsr 1
#include "pspm_search_header.h"

#define TAPES_LOG "/psr/cpsrdata/cpsrtapes.xfer"
#define DLT_LOCK "/tmp/DLT.lock"

class MJD;
class fptm_obs;
class SystemTime;

// these function return some parameter that must be derived from the
// fields in a PSPM_SEARCH_HEADER struct.  Most others are trivially
// obtained directly from the fields

MJD    PSPMstart_time (const PSPM_SEARCH_HEADER* header);
double PSPMduration (const PSPM_SEARCH_HEADER* header);
bool   PSPMverify (const PSPM_SEARCH_HEADER* hdr);

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

int xfer_tape (const char* tapedev, vector<int>* filenos,
	       const vector<string>& disks, char* ext,
	       SystemTime* clock=NULL, int obstype=-1, int keep=1);

#endif // __CPSR_H
