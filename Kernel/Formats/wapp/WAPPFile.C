/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/WAPPFile.h"
#include "Error.h"

// from sigproc-2.4
#include "wapp_header.h"
#include "key.h"

using namespace std;

dsp::WAPPFile::WAPPFile (const char* filename, const char* headername)
  : BlockFile ("WAPP")
{
}

dsp::WAPPFile::~WAPPFile ( )
{
}


bool dsp::WAPPFile::is_valid (const char* filename, int) const
{
  struct HEADERP *h = head_parse( filename );

  if (!h)
    return false;

  close_parse( h );
  return true;
}

void dsp::WAPPFile::open_file (const char* filename)
{
  struct WAPP_HEADER head;
  struct HEADERP *h = head_parse( filename );

  if (!h)
    throw Error (InvalidParam, "dsp::WAPPFile::open_file",
		 "not a WAPP file");

#if 0
  info.set_nbit ();
  info.set_bandwidth ();
  info.set_centre_frequency ();
  info.set_npol ();
  info.set_state ();
  info.set_rate ();
  info.set_start_time ();
  info.set_telescope_code ();
  info.set_source ();

  set_total_samples();
  info.set_default_basis ();

  info.set_mode(string);
#endif

  string prefix="wapp";
  info.set_identifier(prefix+info.get_default_id() );
  info.set_machine("WAPP");	

}

