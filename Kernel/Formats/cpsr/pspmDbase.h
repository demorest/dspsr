//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/pspmDbase.h,v $
   $Revision: 1.3 $
   $Date: 2001/08/01 03:50:01 $
   $Author: wvanstra $ */

#ifndef __pspmDbase_h
#define __pspmDbase_h

#include <vector>
#include <string>
#include "MJD.h"

namespace pspmDbase {

  class entry {

  public:
    int32     scan;        // scan number
    int32     num;         // scan file number

    int32     tape;        // tape number
    int32     file;        // file number

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

    // returns the number of members that match the arguments
    int  match (int32 scan, int32 num, int32 tape, int32 file);

    friend bool operator < (const entry& e1, const entry& e2)
      { return e1.scan < e2.scan || (e1.scan == e2.scan && e1.num < e2.num); }
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

    // returns the database entry that most closely matches the arguments
    // (close = fewest number of bits corrupted in these four header fields)
    entry match (int32 scan, int32 num, int32 tape, int32 file);

  };

}

#endif // __pspmDbase_h
