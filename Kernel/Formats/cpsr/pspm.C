#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <values.h>

#define cpsr 1

#include "pspm_search_header.h"
#include "machine_endian.h"
#include "MJD.h"

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

  toBigEndian (&(header->BACKEND_TYPE), sizeof (int32));
  toBigEndian (&(header->UPDATE_DONE),  sizeof (int32));
  toBigEndian (&(header->HEADER_TYPE),  sizeof (int32));
}

MJD PSPMstart_time (const PSPM_SEARCH_HEADER* header)
{
  MJD start_time (header->mjd_start);

  // MODIFY THE MJD BY THE SCAN FILE NUMBER

  double seconds_per_file = (32768.0 * 32768.0) * header->samp_rate / 1e6;

  for (int file=1; file<header->scan_file_number; file++)
    start_time = start_time + seconds_per_file;

  return start_time;
}

double PSPMduration (const PSPM_SEARCH_HEADER* hdr)
{
  double npts = hdr->file_size * (BITSPERBYTE/(hdr->bit_mode * hdr->num_chans));
  return npts * hdr->samp_rate / 1e6;
}

bool PSPMverify (const PSPM_SEARCH_HEADER* hdr)
{
  return ( (hdr->header_version > 0) &&
	   (hdr->scan_file_number > 0) &&
           (hdr->bit_mode > 0) &&
           (hdr->scan_num > 0) &&
           (hdr->num_chans > 0) &&
           (hdr->file_size > 32768) &&
           (hdr->tape_num > 0) &&
           (hdr->tape_file_number > 0) );
}

