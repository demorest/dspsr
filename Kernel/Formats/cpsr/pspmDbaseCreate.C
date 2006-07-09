/***************************************************************************
 *
 *   Copyright (C) 2001 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "pspmDbase.h"

#include <iostream>

using namespace std;

int usage ()
{
  cerr <<
    "\npspmDbaseCreate - makes CPSR header log file in 'log.txt'\n"
    "\n"
    "first argument should be wildcarded path to headers, such as in:\n\n"
    "pspmDbaseCreate \"/caltech/cpsr.data/search/header/*/*.cpsr\"\n"
       << endl;
  return -1;
}

int main (int argc, char** argv) try {

  if (argc<2 || string(argv[1])=="-h") {
    return usage ();
  }

  pspmDbase::server dbase;

  if (argc==3 && string(argv[1])=="-r")  {
    cerr << "pspmDbaseCreate loading ascii from '" << argv[2] << "'" << endl;
    dbase.load (argv[2]);
  }
  else
    dbase.create(argv[1]);

  cerr << "pspmDbaseCreate " << dbase.entries.size() << " entries" << endl;

  dbase.unload ("hdrlog.txt");
  return 0;
}
catch (string err) {
  cerr << "pspmDbaseCreate error: '" << err << "'" << endl;
  return -1;
}

