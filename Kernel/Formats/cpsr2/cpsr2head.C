/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <vector>
#include <string>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "Error.h"
#include "cpsr2_utils.h"
#include "dirutil.h"

using namespace std;

char* args = "hlk:m:";

void usage();
void parse_it(int argc, char** argv,
	      vector<string>& files, vector<string>& keys,
	      bool& print_unknown);

int main(int argc,char** argv){
  Error::verbose = true;

  unsigned files_done = 0;
  bool print_unknown = false;  // prints 'UNKNOWN' if key not found

  try {

  vector<string> files;
  vector<string> keys;

  parse_it( argc, argv, files, keys, print_unknown);

  for( unsigned ifile=0; ifile<files.size(); ifile++){ try {
    if( !file_exists( files[ifile].c_str() ) ){
      fprintf(stderr,"File '%s' not found!\n",files[ifile].c_str());
      continue;
    }
    string hdr = get_header(files[ifile]);
    
    cout << files[ifile] << "\t";
    
    for( unsigned ikey=0; ikey<keys.size();ikey++){
      string result = scan_header(hdr,keys[ikey],print_unknown);
      cout << result << "\t";
    }
    cout << endl;

    files_done++;
  } catch(Error er) {
    fprintf(stderr,"Caught Error for file '%s':\n",files[ifile].c_str());
    cerr << er << endl;
    continue;
  }

  }



  } catch(Error& er) {
    cerr << er << endl;
    exit(-1);
  } catch( ... ) {
    cerr << "Unknown exception caught\n";
    exit(-1);
  }

  exit(files_done);
}

void usage(){
  cerr << "cpsr2head- scans headers of cpsr2 files\n";

  cerr << "Usage: cpsr2head -[" << args << "] file1 file2...\n";
  cerr << " h                 This help page\n"
       << " k keyword         Keyword [req] ('-k' can be used multiple times)\n"
       << " l                 Print 'UNKNOWN' in field if keyword not found\n"
       << " m metafile        Metafile to parse [no metafile] ('-m' or files required)\n"
       << endl;
  exit(0);
}

void parse_it(int argc, char** argv,
	      vector<string>& files, vector<string>& keys,
	      bool& print_unknown){
  if( argc==1 )
    usage();

  int c;

  while((c = getopt(argc, argv, args)) != -1){
    switch (c) {
      
    case 'h':
      usage();
    case 'k':
      keys.push_back( optarg );
      break;
    case 'l':
      print_unknown = true;
      break;
    case 'm':
      cerr << "-m disabled out of laziness (please contact Willem)" << endl;
      exit(-1);
      // parse_metafile(files,optarg);
      break;
    default:
      throw Error(InvalidParam,"parse_it()",
		  "Unknown parameter '%c'",c);
    }
  }
  
  for (int ai=optind; ai<argc; ai++)
    dirglob (&files, argv[ai]);
  
  if( files.empty() ){
    fprintf(stderr,"You must specify at least 1 filename\n");
    usage();
  }

  if( keys.empty() ){
    fprintf(stderr,"You must specify at least 1 keyword using '-k'\n");
    usage();
  }

}
















