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
#include "dsp/IOManager.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/TwoBitStatsPlotter.h"
#include "Error.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

#define BLK_SIZE 65536     // Size of blocks to read in from file

int main(int argc, char** argv)
{

  dsp::Observation::verbose = true;

  try {

  if(argc!=2)
    {
      perror("Invalid arguements : lbadr64tats <file.lba>\n");
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

  dsp::TimeSeries* dspts = new dsp::TimeSeries();
  dsp::IOManager* manager = new dsp::IOManager;

  manager->set_output (dspts);

  manager->open(filename);

  manager->get_input()->set_block_size ( BLK_SIZE );

  dsp::TwoBitCorrection* unpacker;
  unpacker = dynamic_cast<dsp::TwoBitCorrection*> ( manager->get_unpacker() );

  // Go to work
  int count;
  for(count = 0; count < max; count++)  // Load and unpack (BLK_SIZE*max) bytes
    { 
      manager->operate();
    }

  // Plot Histogram
  cpgopen("?");
  cpglab("","","LBADR 64 MHz Sampler Statistics");
  cpgsvp(0.1,0.9,0.1,0.9);
  dsp::TwoBitStatsPlotter* plot = new dsp::TwoBitStatsPlotter;
  plot->set_data(unpacker);
  plot->plot();
  cpgclos();

  // Free memory
  delete unpacker;

  }
  catch (Error& error)  {
    std::cerr << error << std::endl;
  }

}
