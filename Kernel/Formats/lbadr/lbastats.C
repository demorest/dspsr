/***************************************************************************
 *
 *   Copyright (C) 2005 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/************************************************************************
 ***       SMRO Sampler Statistic Plotter - T. Dolley - 2005          ***
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <cpgplot.h>
#include "dsp/TwoBitStatsPlotter.h"
#include "dsp/SMROTwoBitCorrection.h"
#include "dsp/SMROFile.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define BLK_SIZE 65536     // Size of blocks to read in from file

int main(int argc, char** argv)
{
  if(argc!=2)
    {
      perror("Invalid arguements : SMROStats <file.lba>\n");
      exit(1);
    }

  // Get the filename from the command line
  char* filename = argv[1];

  // Get stats for input file
  struct stat file_stats;
  if(stat(filename,&file_stats))
    {
      perror("Couldn't retrieve file information.\n");
      exit(1);
    }

  // Check length of input file
  int max=1024; // No. of blocks to read. When max=1024;64MB of input used 
  if(file_stats.st_size<(BLK_SIZE*max))
    max=file_stats.st_size/BLK_SIZE;

  // Data loader
  dsp::SMROFile* loader = new dsp::SMROFile;
  loader->open( filename );
  loader->set_output( new dsp::BitSeries );
  loader->set_block_size( BLK_SIZE );

  // Convert to a dsp::TimeSeries
  dsp::SMROTwoBitCorrection* unpacker = new dsp::SMROTwoBitCorrection;
  unpacker->set_input( loader->get_output() );
  unpacker->set_output( new dsp::TimeSeries );

  // Go to work
  int count;
  for(count=0;count<max;count++)  // Load and unpack (BLK_SIZE*max) bytes
    {
      loader->operate();
      unpacker->operate();
    }
  loader->close();

  // Plot Histogram
  cpgopen("?");
  cpglab("","","SMRO Sampler Statistics");
  cpgsvp(0.1,0.9,0.1,0.9);
  dsp::TwoBitStatsPlotter* plot = new dsp::TwoBitStatsPlotter;
  plot->set_data(unpacker);
  plot->plot();
  cpgclos();

  // Free memory
  delete loader,unpacker;
}
