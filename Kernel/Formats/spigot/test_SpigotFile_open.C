#include "dsp/SpigotFile.h"

#include <stdio.h>
#include <iostream>

void usage ()
{
  cerr << "tests opening and parsing spigot files" << endl;
}

int main (int argc, char** argv)
{
  char* metafile = 0;
  bool verbose = false;

  int c;
  while ((c = getopt(argc, argv, "hM:v")) != -1)
    switch (c) {

    case 'h':
      usage ();
      return 0;

    case 'M':
      metafile = optarg;
      break;

    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector<string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return 0;
  }

  dsp::SpigotFile file;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) {

    if ( !file.is_valid( filename ) )
      cerr << "test: " << filename << " not a Spigot file" << endl;

    file.open( filename );

    cerr << "test: " << filename << " opened" << endl;

    file.obs2file (stderr);

  }

}


