/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/S2TwoBitCorrection.h"
#include "dsp/S2TwoBitTable.h"
#include "dsp/Observation.h"
#include "Error.h"

using namespace std;

bool dsp::S2TwoBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "S2" && observation->get_nbit()==2;
}


/*!
  The VLBA encode offset binary;  the AT encode sign magnitude
*/
dsp::S2TwoBitCorrection::S2TwoBitCorrection (char telescope)
  : TwoBitCorrection ("S2TwoBitCorrection")
{
  resynch_period = resynch_start = resynch_end = 0.0;

  match (telescope);
}

void dsp::S2TwoBitCorrection::match (const Observation* observation)
{
  match ( Tempo::code(observation->get_telescope()) );
}

void dsp::S2TwoBitCorrection::match (char telescope)
{
  switch ( telescope ) {

  case '7':
    if (verbose)
      cerr << "dsp::S2TwoBitCorrection::match Parkes (AT)" << endl;
    resynch_period = 10.0;
    resynch_start = 9.949;
    resynch_end = 0.0001;
    
    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    break;
    
  case '2':
    if (verbose)
      cerr << "dsp::S2TwoBitCorrection::match ATCA (AT)" << endl;
    resynch_period = 10.0;
    resynch_start = 9.940;
    resynch_end = 0.15;
    
    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    break;
    
  case '6':

    if (verbose)
      cerr << "dsp::S2TwoBitCorrection::match Tidbinbilla (AT)" << endl;
    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    // CHANGED BY CWEST - noone was using the VLBA mode, and I wanted AT mode.
    //    if (verbose)
    //      cerr << "dsp::S2TwoBitCorrection::match Tidbinbilla (VLBA)" << endl;
    //    table = new S2TwoBitTable (TwoBitTable::OffsetBinary);
    break;

  case '3':
    if (verbose)
      cerr << "dsp::S2TwoBitCorrection::match Arecibo (VLBA)" << endl;
    table = new S2TwoBitTable (TwoBitTable::OffsetBinary);
    break;
    
  case '4':
    if (verbose)
      cerr << "dsp::S2TwoBitCorrection::match Hobart (AT)" << endl;
    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    break;

  default:
    throw Error (InvalidParam, "S2TwoBitCorrection::match",
		 "unknown telescope = %c", telescope );
  }
  
   
}


void dsp::S2TwoBitCorrection::unpack ()
{
  TwoBitCorrection::unpack ();

  // at some point in the near future, this function should also NULL
  // out the data recorded during the 10s synch of the S2-DAS

  if (resynch_period == 0.0)
    return;

#if 0
  for (MJD
  double start_seconds = fmod (fs->start_time.fracday()*86400.0, 
			       PKS_RESYNCH_PERIOD);

  double incr_seconds = fs->get_ppweight()/fs->rate;
  
  // If (MJD>so_and_so) check....
  if(resynctype == s2_2bit_correct::ON && packtype == s2_2bit_correct::AT)
    {
      if (fs->telescope == TELID_PKS) {  // Parkes
	if(start_seconds < PKS_RESYNCH_END) n_in = 0;
	if(start_seconds+incr_seconds > PKS_RESYNCH_START) n_in = 0;
      }
      else if (fs->telescope == TELID_ATCA) {
	if(start_seconds < ATCA_RESYNCH_END) n_in = 0;
	if(start_seconds+incr_seconds > ATCA_RESYNCH_START) n_in = 0;
      }
    }

      
      start_seconds += (double)fs->get_ppweight()/fs->rate;
      start_seconds = fmod (start_seconds, PKS_RESYNCH_PERIOD);

#endif

}
