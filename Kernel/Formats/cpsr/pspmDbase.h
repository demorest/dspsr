//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2001 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr/pspmDbase.h

#ifndef __pspmDbase_h
#define __pspmDbase_h

#include <vector>
#include <string>

#include "environ.h"
#include "MJD.h"

namespace pspmDbase {

  class entry {

  public:
    int32_t       scan;      // scan number
    int32_t       num;       // scan file number

    int32_t       tape;      // tape number
    int32_t       file;      // file number

    MJD         start;     // MJD of start time
    int         ttelid;    // tempo telescope code
    std::string name;      // name of source
    double    frequency;   // MHz
    double    bandwidth;   // MHz (-ve for lower sideband)
    double    tsamp;       // sampling period in microseconds
    int       ndigchan;    // number of digitizer channels
    int       nbit;        // number of bits per sample
    int64_t     ndat;        // number of time samples

    double    duration()   // in seconds
      { return double(ndat) * tsamp * 1e-6; }

    // parse from database line
    entry (const char* str) { load(str); }
    // null construct
    entry ();

    // load from ascii string
    void load (const char* str);
    // unload ascii string
    void unload (std::string& str);

    // create from PSPM_SEARCH_HEADER
    void create (void* hdr);

    // returns the number of members that match the arguments
    int  match (int32_t scan, int32_t num, int32_t tape, int32_t file);

    // returns true on exact match
    bool match (int32_t _tape, int32_t _file)
      {  return tape==_tape && file==_file; }

    // used for sorting. by scan number, then by scan file number
    friend bool operator < (const entry& e1, const entry& e2)
      { return e1.scan < e2.scan || (e1.scan == e2.scan && e1.num < e2.num); }

    std::string tapename ();   // returns CPSR1234
    std::string identifier (); // returns CPSR1234.32
  };

  // returns an entry from the default database
  entry Entry (void* hdr);

  class server {

  public:
    // light-weight on RAM (internal=false), or 
    // light-weight on NFS (internal=true)
    bool internal;
    std::vector<entry> entries;

    server () { internal = true; }
    ~server () {}

    // server::create - uses dirglob to expand wild-card-style
    // list of files containing CPSR headers 
    // (such as /caltech/cpsr.data/search/header/*/*.cpsr on orion)
    void create (const char* glob);

    // loads ascii version from file
    void load (const char* dbase_filename = default_name());
    // unloads ascii version to file
    void unload (const char* dbase_filename);

    //
    // Returns the database entry that most closely matches the arguments
    // (close = fewest number of bits corrupted in these four header fields)
    //
    // Use this interface when dealing with information taken from
    // a PSPM_SEARCH_HEADER, so that all four fields may be used to
    // make a positive match
    entry match (int32_t scan, int32_t num, int32_t tape, int32_t file);

    //
    // Returns the database entry for CPSRtape.file
    //
    // Use this interface when you positively know the tape and file
    entry match (int32_t tape, int32_t file);

    // convenience. return match from PSPM_SEARCH_HEADER
    entry match (void* hdr);

    // returns the default name of the database
    static const char* default_name();

  };

}

#endif // __pspmDbase_h
