#include "S2TwoBitCorrection.h"
#include "S2TwoBitTable.h"

#include "genutil.h"

/*!
  The VLBA encode offset binary;  the AT encode sign magnitude
*/
dsp::S2TwoBitCorrection::S2TwoBitCorrection (char telescope)
  : TwoBitCorrection ("S2TwoBitCorrection")
{
  nchannel = 2;
  channels_per_byte = 1;

  resynch_period = resynch_start = resynch_end = 0.0;

  switch (telescope) {

  case Telescope::Parkes:
    resynch_period = 10.0;
    resynch_start = 9.949;
    resynch_end = 0.0001;
    
    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    break;

  case Telescope::ATCA:
    resynch_period = 10.0;
    resynch_start = 9.940;
    resynch_end = 0.15;

    table = new S2TwoBitTable (TwoBitTable::SignMagnitude);
    break;

  case Telescope::Tidbinbilla:
  case Telescope::Arecibo:
    table = new S2TwoBitTable (TwoBitTable::OffsetBinary);

  default:
    throw_str ("S2TwoBitCorrection:: unknown telescope = %c", telescope);

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
