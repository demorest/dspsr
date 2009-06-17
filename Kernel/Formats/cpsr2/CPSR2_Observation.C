/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/CPSR2_Observation.h"
#include "cpsr2_header.h"

#include "strutil.h"
#include "Types.h"
#include "coord.h"

using namespace std;

dsp::CPSR2_Observation::CPSR2_Observation (const char* header)
{
  hdr_version = "CPSR2_HEADER_VERSION";

  if (!header)
    return;

  load (header);

  if ( get_nchan() > 1 && get_npol() == 1)
    set_state (Signal::Intensity);
  
#if 0

  if (version < 0.2) {
    // //////////////////////////////////////////////////////////////////////
    //
    // NMBYTES
    //
    uint64_t offset_Mbytes = 0;
    if (ascii_header_get (header, "NMBYTES", UI64, &offset_Mbytes) < 0)
      cerr << "CPSR2_Observation - no NMBYTES...  assuming local" << endl;

    cerr << "CPSR2_HEADER_VERSION 0.1 offset MBytes " << offset_Mbytes << endl;

    uint64_t MByte = 1024 * 1024;
    offset_bytes += offset_Mbytes * MByte;
  }

#endif

  char hdrstr[64];

  // //////////////////////////////////////////////////////////////////////
  //
  // PRIMARY
  //
  if (ascii_header_get (header, "PRIMARY", "%s", hdrstr) < 0)
    throw Error (InvalidState, "CPSR2_Observation", "failed read PRIMARY");

  string primary = hdrstr;
  prefix = "u";

  if (primary == "cpsr1")
    prefix = "m";
  if (primary == "cpsr2")
    prefix = "n";

  if (primary == "cgsr1")
    prefix = "p";
  if (primary == "cgsr2")
    prefix = "q";

  // make an identifier name
  set_mode (stringprintf ("%d-bit mode", get_nbit()));
  set_machine ("CPSR2");

}
