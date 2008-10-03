#include "dsp/BandpassMonitor.h"
#include "dsp/Rescale.h"

#include "Error.h" 

using namespace std;

dsp::BandpassMonitor::BandpassMonitor()
{
}

void dsp::BandpassMonitor::output_state (Rescale* rescale)
{
  MJD epoch = rescale->get_update_epoch();

  char datestr [64];
  std::string stamp = epoch.datestr (datestr, 64, "%Y-%m-%d-%H:%M:%S");

  unsigned npol  = rescale->get_input()->get_npol();
  unsigned nchan = rescale->get_input()->get_nchan();
  unsigned ndat  = rescale->get_nsample();

  rms.resize (nchan);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    const float* data = 0;

    // the r.m.s. is one over the scale
    const float* scale = rescale->get_scale(ipol);
    for (unsigned ichan=0; ichan < nchan; ichan++)
      rms[ichan] = 1.0/scale[ichan];

    data = &(rms[0]);
    dump (stamp, ipol, nchan, data, ".bps");

    data = rescale->get_offset(ipol);
    dump (stamp, ipol, nchan, data, ".bp");
    
    data = rescale->get_time(ipol);
    dump (stamp, ipol, ndat, data, ".ts");
  }
}

void dsp::BandpassMonitor::dump (const string& timestamp, 
				 unsigned pol, unsigned ndat, 
				 const float* data, const char* ext)
{
  string filename = timestamp + ext + tostring(pol);
  string temp_filename = filename + ".tmp";

  FILE* fptr = fopen( temp_filename.c_str(), "wb" );
  if (!fptr)
    throw Error (FailedSys, "dsp::BandpassMonitor::dump",
		 "fopen (" + temp_filename + ", \"wb\")");

  fwrite (data, ndat, sizeof(float), fptr);
  fclose (fptr);

  rename (temp_filename.c_str(), filename.c_str());
}
