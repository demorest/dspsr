/***************************************************************************
 *
 *   Copyright (C) 2012 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CommandLineHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

std::string dsp::CommandLineHeader::convert (int argc, char **argv, 
    std::string filename) 
{

  FILE *hdrfile=NULL;

  if (filename.empty()) 
  {
    // Generate a temp file
    char tmpfname[64] = "/tmp/dspsrhdrXXXXXX";
    int fd = mkstemp(tmpfname);
    if (fd == -1) 
      throw Error(FailedSys, "dsp::CommandLineHeader::convert", "mkstemp");
    filename = tmpfname;
    hdrfile = fdopen(fd, "w");
    if (hdrfile==NULL)
      throw Error(FailedSys, "dsp::CommandLineHeader::convert", "fdopen");
  }
  else 
  {
    // Open the requested file
    hdrfile = fopen(filename.c_str(),"w");
    if (hdrfile==NULL)
      throw Error(FailedSys, "dsp::CommandLineHeader::convert", 
          "fopen(%s)", filename.c_str());
  }

  // Convert the args, assume each argv entry is a new line, 
  // and change first '=' detected into whitespace.  Could probably
  // tweak argv values in place...
  for (int ai=optind; ai<argc; ai++) 
  {
    char line[128];
    strncpy(line, argv[ai], 127);
    line[127] = '\0';
    char *eq = strchr(line, '=');
    if (eq!=NULL) { *eq = ' '; }
    fprintf(hdrfile, "%s\n", line);
  }

  // Close file, return name
  fclose(hdrfile);
  return filename;
}
