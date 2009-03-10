/***************************************************************************
 *
 *   Copyright (C) 2001 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "pspmDbase.h"
#include "dirutil.h"
#include "pspm++.h"
#include "Error.h"

// CPSR hdr and unpacking routines
#define cpsr 1
#include "pspm_search_header.h"

#include <algorithm>
#include <stdio.h>
#include <string.h>

using namespace std;

// #define _DEBUG 1

/*
 * CPSR Sideband modes copied from pspm.h
 */
#define UNKNOWN_SIDEBAND	0
#define	SSB_LOWER		1
#define	SSB_UPPER		2
#define	DSB_SKYFREQ		3
#define	DSB_REVERSED		4


static string default_fname;
static pspmDbase::server default_server;

const char* pspmDbase::server::default_name()
{
  if (default_fname.empty()) {
    char* psrhome = getenv ("PSRHOME");
    if (!psrhome)
      throw Error (InvalidState, "pspmDbase::server::default",
		   "PSRHOME not defined");
    default_fname = psrhome;
    default_fname += "/runtime/cpsr/header.entries";
  }
  return default_fname.c_str();
}

pspmDbase::entry pspmDbase::Entry (void* hdr)
{
  if (default_server.entries.size() == 0)
    default_server.load();   // load from the default location

  return default_server.match (hdr);
}

void ascii_dump (const PSPM_SEARCH_HEADER* hdr);

pspmDbase::entry::entry ()
{
  scan = num = tape = file = -1;
  ttelid = ndigchan = nbit = -1;
  ndat = -1;
  frequency = bandwidth = tsamp = 0;
}

// create from PSPM_SEARCH_HEADER
void pspmDbase::entry::create (void* vhdr)
{
  PSPM_SEARCH_HEADER* hdr = (PSPM_SEARCH_HEADER*) vhdr;
#if _DEBUG
  ascii_dump (hdr);
#endif

  // set the field values
  scan      = hdr->scan_num;
  num       = hdr->scan_file_number;
  tape      = hdr->tape_num;
  file      = hdr->tape_file_number;
  
  start     = PSPMstart_time (hdr);
  
  ttelid    = hdr->observatory;
  name      = hdr->psr_name;
  frequency = hdr->rf_freq;
  bandwidth = hdr->bw;
  if ((hdr->SIDEBAND == SSB_LOWER) || (hdr->SIDEBAND == DSB_REVERSED))
    bandwidth *= -1.0;
  
  tsamp     = hdr->samp_rate;
  ndigchan  = hdr->num_chans;
  nbit      = hdr->bit_mode;
  
  int64 fsize;
  if (hdr->ll_file_size == 0)
    // old style - pre-August 1999
    fsize = hdr->file_size;
  else
    fsize = hdr->ll_file_size;
  
  ndat = 8*(fsize/(ndigchan*nbit));        // number of time samples
}


// load from ascii string
static char buffer[256];
static char mjdstr[32];
static char src[32];

void pspmDbase::entry::load (const char* str) 
{
  int s = sscanf (str,
		     I32" "I32" "I32" "I32
		     " %s %s"
		     " %lf %lf %lf"
		     " %d %d %d "I64,
		     &scan, &num, &tape, &file,
		     mjdstr, src, 
		     &frequency, &bandwidth, &tsamp,
		     &ttelid, &ndigchan, &nbit, &ndat);

  if (s != 13)
    throw Error (InvalidParam, "pspmDbase::entry::load",
		 "error parsing '%s'", str);

  start.Construct(mjdstr);
  name = src;
}

// unload ascii string
void pspmDbase::entry::unload (string& str)
{
  strcpy (mjdstr, start.printdays(15).c_str());
  int s = sprintf (buffer, 
		      I32" "I32" "I32" "I32
		      " %s %s"
		      " %lf %lf %lf"
		      " %d %d %d "I64,
		      scan, num, tape, file,
		      mjdstr, name.c_str(), 
		      frequency, bandwidth, tsamp,
		      ttelid, ndigchan, nbit, ndat );
  str = buffer;
}

int pspmDbase::entry::match (int32 _scan, int32 _num, int32 _tape, int32 _file)
{
  int matches = 0;
  if (scan == _scan)
    matches ++;
  if (num == _num)
    matches ++;
  if (tape == _tape)
    matches ++;
  if (file == _file)
    matches ++;

  return matches;
}

string pspmDbase::entry::tapename ()
{
  sprintf (src, "CPSR%04d", tape);
  return string (src);
}

string pspmDbase::entry::identifier ()
{
  sprintf (src, "CPSR%04d.%d", tape, file);
  return string (src);
}

// server::create - uses dirglob to expand wild-card-style
// list of files containing CPSR headers 
// (such as /caltech/cpsr.data/search/header/*/*.cpsr on orion)
void pspmDbase::server::create (const char* glob)
{
  internal = true;
  entries.clear();

  vector<string> filenames;
  dirglob (&filenames, glob);

  entry next;

  for (unsigned ifile=0; ifile<filenames.size(); ifile++) {
    PSPM_SEARCH_HEADER* hdr = pspm_read (filenames[ifile].c_str());

    if (!hdr) throw Error (FailedCall, "pspmDbase::server::create",
			   "failed pspm_read(" + filenames[ifile] + ")");

    try { next.create(hdr); }
    catch (...) { continue; }

    entries.push_back(next);
  }

  sort (entries.begin(), entries.end());
}

// loads ascii version from file
void pspmDbase::server::load (const char* dbase_filename)
{
  if (!internal)
    fprintf (stderr, "pspmDbase::server::load only internal implemented\n");
  internal = true;

  FILE* fptr = fopen (dbase_filename, "r");
  if (!fptr)
    throw Error (FailedSys, "pspmDbase::server::load",
		 "error fopen(%s)\n", dbase_filename);

  entries.clear();
  entry next;

  while ( fgets (buffer, 256, fptr) ) {
    try { next.load(buffer); }
    catch (...) { continue; }
    entries.push_back(next);
  }
  fclose (fptr);
}

// unloads ascii version to file
void pspmDbase::server::unload (const char* dbase_filename)
{
  if (!internal)
    return;

  FILE* fptr = fopen (dbase_filename, "w");
  if (!fptr)
    throw Error (FailedSys, "pspmDbase::server::unload",
		 "error fopen(%s)\n", dbase_filename);

  string out;
  for (unsigned ie=0; ie<entries.size(); ie++) {
    entries[ie].unload(out);
    fprintf (fptr, "%s\n", out.c_str());
  }
  fclose (fptr);
}

pspmDbase::entry pspmDbase::server::match (void* vhdr)
{
  PSPM_SEARCH_HEADER* hdr = (PSPM_SEARCH_HEADER*) vhdr;

  pspmDbase::entry result;

  try {
    result = match (hdr->scan_num, hdr->scan_file_number,
		    hdr->tape_num, hdr->tape_file_number);
  }
  catch (string xcptn) {
    cerr << xcptn << endl;
    cerr << "pspmDbase::server::match hope for the best" << endl;
    result.create (hdr);
  }
  return result;
}

pspmDbase::entry pspmDbase::server::match (int32 tape, int32 file)
{
  for (unsigned ie=0; ie<entries.size(); ie++)
    if (entries[ie].match( tape, file ))
      return entries[ie];

  throw Error (InvalidState, "pspmDbase::server::match", "none found");
}

pspmDbase::entry
pspmDbase::server::match (int32 scan, int32 num, int32 tape, int32 file)
{
  int ties=0;
  int best=0;

  unsigned index=0;

  for (unsigned ie=0; ie<entries.size(); ie++) {
    int matches = entries[ie].match( scan, num, tape, file );
    if (matches > best) {
      best = matches;
      index = ie;
      ties = 0;
    }
    else if (matches == best)
      ties ++;
  }

  if (best < 3)
    throw Error (InvalidState, "pspmDbase::server::match", "not found");

  if (best == 3) {
    cerr << "pspmDbase::server::match partial with " <<ties<< " ties" << endl;
    if (ties) {
      cerr << "pspmDbase::server::match WILLEM, CODE A SOLUTION!" << endl;
      throw Error (InvalidState, "pspmDbase::server::match", "TIED");
    }
  }

  return entries[index];
}

void ascii_dump (const PSPM_SEARCH_HEADER* hdr)
{

  fprintf (stderr, "File size:       %ld\n", hdr->file_size);
  fprintf (stderr, "Large File Size: "I64"\n", hdr->ll_file_size);
  fprintf (stderr, "Large Offset:    "I64"\n", hdr->ll_file_offset);
  fprintf (stderr, "MJD in hdr    %40.38Lf\n", hdr->mjd_start);

  fprintf (stderr, "tick offset:     %30.28lf\n\n", hdr->tick_offset);
  fprintf (stderr, "tape_num:        %d\n", hdr->tape_num);
  fprintf (stderr, "tape_file_number:%d\n", hdr->tape_file_number);
  fprintf (stderr, "scan_num:        %d\n", hdr->scan_num);
  fprintf (stderr, "scan_file_number %d\n", hdr->scan_file_number);
  fprintf (stderr, "file_size        %d\n\n", hdr->file_size);

  fprintf (stderr, "LMST in hdr   %lf\n", hdr->pasmon_lmst);

  fprintf (stderr, "Pulsar:          %s\n",  hdr->psr_name);
  fprintf (stderr, "Date:            %s\n",  hdr->date);
  fprintf (stderr, "Start Time:      %s\n",  hdr->start_time);

  fprintf (stderr, "pasmon_daynumber:%d\n",  hdr->pasmon_daynumber);
  fprintf (stderr, "pasmon_ast:      %d\n",  hdr->pasmon_ast);
 
  fprintf (stderr, "Centre Freq:     %lf\n", hdr->rf_freq);
  fprintf (stderr, "Sampling Period: %lf\n", hdr->samp_rate);
  fprintf (stderr, "Bandwidth:       %lf\n", hdr->bw);
  fprintf (stderr, "SIDEBAND:        %d:",  hdr->SIDEBAND);
  switch (hdr->SIDEBAND) 
    {
    case UNKNOWN_SIDEBAND:
      fprintf (stderr, "Unknown (assume USB)\n");
      break;
    case SSB_LOWER:
      fprintf (stderr, "SSB Lower\n");
      break;
    case SSB_UPPER:
      fprintf (stderr, "SSB Upper\n");
      break;
    case DSB_SKYFREQ:
      fprintf (stderr, "DSB Sky frequency order\n");
      break;
    case DSB_REVERSED:
      fprintf (stderr, "DSB Sky reversed frequency order\n");
      break;
    default:
      fprintf (stderr, "Internal error\n");
      break;
    }
  
  fprintf (stderr, "Telescope:       %d\n",  hdr->observatory);
  fprintf (stderr, "Channels:        %ld\n", hdr->num_chans);
  fprintf (stderr, "Bit Mode:        %ld\n", hdr->bit_mode);
  
  fflush (stderr);
}
