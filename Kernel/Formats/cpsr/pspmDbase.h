//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/pspmDbase.h,v $
   $Revision: 1.1 $
   $Date: 2001/07/31 20:26:22 $
   $Author: wvanstra $ */

#ifndef __pspmDbase_h
#define __pspmDbase_h

#include <vector>
#include <string>
#include "MJD.h"

namespace pspmDbase {

  class entry {

  public:
    int       scan;        // scan number
    int       num;         // scan file number

    int       tape;        // tape number
    int       file;        // file number

    MJD       start;       // MJD of start time
    int       ttelid;      // tempo telescope code
    string    name;        // name of source
    double    frequency;   // MHz
    double    bandwidth;   // MHz (-ve for lower sideband)
    double    tsamp;       // sampling period in microseconds
    int       ndigchan;    // number of digitizer channels
    int       nbit;        // number of bits per sample
    int64     ndat;        // number of time samples

    // parse from database line
    entry (const char* str) { load(str); }
    // null construct
    entry ();

    // load from ascii string
    void load (const char* str);
    // unload ascii string
    void unload (string& str);

    // create from PSPM_SEARCH_HEADER
    void create (void* hdr);

  };


  class server {

  public:
    // light-weight on RAM (internal=false), or 
    // light-weight on NFS (internal=true)
    bool internal;
    vector<entry> entries;

    server () { internal = true; }
    ~server () {}

    // server::create - uses dirglob to expand wild-card-style
    // list of files containing CPSR headers 
    // (such as /caltech/cpsr.data/search/header/*/*.cpsr on orion)
    void create (const char* glob);

    // loads ascii version from file
    void load (const char* dbase_filename);
    // unloads ascii version to file
    void unload (const char* dbase_filename);

  };

}

#endif // __pspmDbase_h
