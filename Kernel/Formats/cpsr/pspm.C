/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <values.h>
#include <math.h>

#define cpsr 1

#include "pspm_search_header.h"
#include "machine_endian.h"
#include "pspm++.h"
#include "genutil.h"

// #if (sizeof(PSPM_SEARCH_HEADER) != PSPM_HEADER_SIZES)
// #error Architecture Error! PSPM header size is invalid.
// #endif

int PSPMdisplay (FILE* out, PSPM_SEARCH_HEADER* header, const char* field)
{
  fprintf (out, "%s  [f:%lf] CPSR%04d %d\n", header->psr_name, 
		header->rf_freq, header->tape_num, header->tape_file_number);
  return 0;
}


void PSPMfromBigEndian (PSPM_SEARCH_HEADER* header)
{
  fromBigEndian (&(header->samp_rate),   sizeof (double));
  fromBigEndian (&(header->pasmon_az),   sizeof (double));
  fromBigEndian (&(header->pasmon_za),   sizeof (double));
  fromBigEndian (&(header->user_az),     sizeof (double));
  fromBigEndian (&(header->user_za),     sizeof (double));
  fromBigEndian (&(header->pasmon_lmst), sizeof (double));
  fromBigEndian (&(header->rf_freq),     sizeof (double));

  fromBigEndian (&(header->tick_offset), sizeof (double));
  fromBigEndian (&(header->bw),          sizeof (double));
  fromBigEndian (&(header->length_of_integration), sizeof (double));

  fromBigEndian (&(header->header_version),   sizeof (int32));
  fromBigEndian (&(header->scan_file_number), sizeof (int32));
  fromBigEndian (&(header->bit_mode),    sizeof (int32));
  fromBigEndian (&(header->scan_num),    sizeof (int32));
  fromBigEndian (&(header->tc),          sizeof (int32));
  fromBigEndian (&(header->num_chans),   sizeof (int32));

  fromBigEndian (&(header->pasmon_wrap),      sizeof (int32));
  fromBigEndian (&(header->pasmon_feed),      sizeof (int32));
  fromBigEndian (&(header->pasmon_daynumber), sizeof (int32) );
  fromBigEndian (&(header->pasmon_ast),       sizeof (int32));

  fromBigEndian (&(header->file_size),        sizeof (int32));
  fromBigEndian (&(header->tape_num),         sizeof (int32));
  fromBigEndian (&(header->tape_file_number), sizeof (int32));

  fromBigEndian (&(header->user_ra),          sizeof (double));
  fromBigEndian (&(header->user_dec),         sizeof (double));
  fromBigEndian (&(header->chan_first_freq),  sizeof (double));
  fromBigEndian (&(header->chan_spacing),     sizeof (double));

  fromBigEndian (&(header->SIDEBAND),         sizeof (int));
  fromBigEndian (&(header->observatory),      sizeof (int));

  // Do NOT byte-swap the long double MJD, as it is interpretted as big endian

  fromBigEndian (&(header->ll_file_offset),   sizeof (int64));
  fromBigEndian (&(header->ll_file_size),     sizeof (int64));

  fromBigEndian (&(header->BACKEND_TYPE), sizeof (int32));
  fromBigEndian (&(header->UPDATE_DONE),  sizeof (int32));
  fromBigEndian (&(header->HEADER_TYPE),  sizeof (int32));
}

void PSPMtoBigEndian (PSPM_SEARCH_HEADER* header)
{
  toBigEndian (&(header->samp_rate),   sizeof (double));
  toBigEndian (&(header->pasmon_az),   sizeof (double));
  toBigEndian (&(header->pasmon_za),   sizeof (double));
  toBigEndian (&(header->user_az),     sizeof (double));
  toBigEndian (&(header->user_za),     sizeof (double));
  toBigEndian (&(header->pasmon_lmst), sizeof (double));
  toBigEndian (&(header->rf_freq),     sizeof (double));

  toBigEndian (&(header->tick_offset), sizeof (double));
  toBigEndian (&(header->bw),          sizeof (double));
  toBigEndian (&(header->length_of_integration), sizeof (double));

  toBigEndian (&(header->header_version),   sizeof (int32));
  toBigEndian (&(header->scan_file_number), sizeof (int32));
  toBigEndian (&(header->bit_mode),    sizeof (int32));
  toBigEndian (&(header->scan_num),    sizeof (int32));
  toBigEndian (&(header->tc),          sizeof (int32));
  toBigEndian (&(header->num_chans),   sizeof (int32));

  toBigEndian (&(header->pasmon_wrap),      sizeof (int32));
  toBigEndian (&(header->pasmon_feed),      sizeof (int32));
  toBigEndian (&(header->pasmon_daynumber), sizeof (int32) );
  toBigEndian (&(header->pasmon_ast),       sizeof (int32));

  toBigEndian (&(header->file_size),        sizeof (int32));
  toBigEndian (&(header->tape_num),         sizeof (int32));
  toBigEndian (&(header->tape_file_number), sizeof (int32));

  toBigEndian (&(header->user_ra),          sizeof (double));
  toBigEndian (&(header->user_dec),         sizeof (double));
  toBigEndian (&(header->chan_first_freq),  sizeof (double));
  toBigEndian (&(header->chan_spacing),     sizeof (double));

  toBigEndian (&(header->SIDEBAND),         sizeof (int));
  toBigEndian (&(header->observatory),      sizeof (int));

  toBigEndian (&(header->ll_file_offset),   sizeof (int64));
  toBigEndian (&(header->ll_file_size),     sizeof (int64));

  toBigEndian (&(header->BACKEND_TYPE), sizeof (int32));
  toBigEndian (&(header->UPDATE_DONE),  sizeof (int32));
  toBigEndian (&(header->HEADER_TYPE),  sizeof (int32));
}

string PSPMsource (const PSPM_SEARCH_HEADER* header)
{
  return string (header->psr_name);
}

MJD PSPMstart_time (const PSPM_SEARCH_HEADER* header)
{
  MJD start_time (header->mjd_start);

  if (header->ll_file_size == 0) {
    // old style - pre-August 1999
    // MODIFY THE MJD BY THE SCAN FILE NUMBER
    double seconds_per_file = (32768.0 * 32768.0) * header->samp_rate / 1e6;
    start_time += seconds_per_file * double(header->scan_file_number - 1);
  }
  else
    start_time += (double(header->ll_file_offset) / 1e6) * header->samp_rate;

  return start_time;
}

double PSPMduration (const PSPM_SEARCH_HEADER* hdr)
{
  double fsize;

  if (hdr->ll_file_size == 0)
    // old style - pre-August 1999
    fsize = double(hdr->file_size);
  else
    fsize = double(hdr->ll_file_size);
  
  double npts = fsize*double(BITSPERBYTE)/double(hdr->bit_mode*hdr->num_chans);

  return npts * hdr->samp_rate / 1e6;
}


string PSPMidentifier (const PSPM_SEARCH_HEADER* hdr)
{
  char id [20];
  if (hdr->observatory == 7)
    sprintf (id, "CPSR%04d.%d", hdr->tape_num, hdr->tape_file_number);
  else
    sprintf (id, "CBR%04d.%d", hdr->tape_num, hdr->tape_file_number);

  return string(id);
}

bool PSPMverify (PSPM_SEARCH_HEADER* hdr, bool verbose)
{
  if (hdr->header_version < 0) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid header_version:%d\n",
	       (int)hdr->header_version);
    return false;
  }
  if (hdr->ll_file_offset == 0)  {
    if (hdr->scan_file_number < 0) {
      if (verbose)
	fprintf (stderr, "PSPMverify: invalid scan_file_number:%d\n",
		 (int)hdr->scan_file_number);
      return false;
    }
  }
  else if (hdr->ll_file_offset < 0)  {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid ll_file_offset:%ld\n",
	       hdr->ll_file_offset);
    return false;
  }
  if (hdr->bit_mode < 0) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid bit_mode:%d\n",
	       (int)hdr->bit_mode);
    return false;
  }
  if (hdr->num_chans < 0) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid num_chans:%d\n",
	       (int)hdr->num_chans);
    return false;
  }
  if (hdr->file_size < 32768) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid file_size:%d\n",
	       (int)hdr->file_size);
    return false;
  }
  if (hdr->tape_num < 0) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid tape_num:%d\n",
	       (int)hdr->tape_num);
    return false;
  }
  if (hdr->tape_file_number < 0) {
    if (verbose)
      fprintf (stderr, "PSPMverify: invalid tape_file_number:%d\n",
	       (int)hdr->tape_file_number);
    return false;
  }
  
  // some more sanity checks
  if (hdr->ll_file_size) {
    // double check that the file_offset is a multiple of 1MB
    int64 mb = 1024 * 1024;
    if (hdr->ll_file_offset % mb) {
      if (verbose)
	fprintf (stderr, "PSPMverify: offset="I64" corrupted.\n",
		 hdr->ll_file_offset);
      return false;
    }
  }
  
  // what can you do?
  float rate = 1.0 / (hdr->samp_rate);  // samples per microsecond
  float bw = fabs (hdr->bw);      // bandwidth in MHz
  if (bw != rate) {
    if (verbose)
      fprintf (stderr, "PSPMverify: Nyquist mismatch: bw=%f MHz."
	       "  sampling rate=%f\n", bw, rate);
    return false;
  }
  return true;
}

static PSPM_SEARCH_HEADER* static_header = NULL;

PSPM_SEARCH_HEADER* pspm_read (int fd)
{
  unsigned header_size = sizeof(PSPM_SEARCH_HEADER);

  if (static_header == NULL) {
    static_header = (PSPM_SEARCH_HEADER*) malloc (header_size);
    if (static_header == NULL)  {
      fprintf (stderr, "pspm_read: Could not malloc PSPM header (%d bytes)",
	sizeof (PSPM_SEARCH_HEADER));
      throw ("pspm_read: memory error");
    }
  }

  ssize_t to_read = header_size;
  char* buf = (char*)(void*) static_header;
  int retries = 3;
  while (retries && to_read) {
    ssize_t bread = read (fd, (void*) buf, to_read);
    if (bread < 0) {
      perror ("pspm_read::Couldn't read header");
      return NULL;
    }
    to_read -= bread;
    buf += bread;
    retries --;
  }
  if (to_read && !retries)
    return NULL;

  PSPMfromBigEndian (static_header);

  return static_header;
}

PSPM_SEARCH_HEADER* pspm_read (const char* filename)
{
  int datfd = open (filename, O_RDONLY);
  if (datfd < 0) {
    fprintf (stderr, "pspm_read::Couldn't open '%s'\n", filename);
    perror (":");
    return NULL;
  }
  PSPM_SEARCH_HEADER* ret = pspm_read (datfd);
  close (datfd);
  return ret;
}

PSPM_SEARCH_HEADER* pspm_read (const char* tapedev, int filenum) 
{
  int tfd = open (tapedev, O_RDONLY);
  if (tfd < 0) {
    fprintf (stderr, "pspm_read:: Could not open '%s'", tapedev);
    perror (":");
    return NULL;
  }

  if (tapepos (tfd, filenum-1) < 0) {
    fprintf (stderr, "pspm_read:: error positioning tape:'%s' to file:%d\n",
	     tapedev, filenum);
    close (tfd);
    return NULL;
  }

  PSPM_SEARCH_HEADER* ret = pspm_read (tfd);

  // re-position the tape and get it ready for dd
  if (tapepos (tfd, filenum-1) < 0) {
    fprintf (stderr, "pspm_read:: error positioning tape:'%s' to file:%d\n",
	     tapedev, filenum);
    close (tfd);
    return NULL;
  }
  close (tfd);
  return ret;
}
