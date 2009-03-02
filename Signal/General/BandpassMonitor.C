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

  temp.resize (nchan);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    const float* data = &(temp[0]);

    const double* mean = rescale->get_mean (ipol);
    for (unsigned ichan=0; ichan < nchan; ichan++)
      temp[ichan] = mean[ichan];
    dump (stamp, ipol, nchan, data, ".bp");

    const double* variance = rescale->get_variance (ipol);
    for (unsigned ichan=0; ichan < nchan; ichan++)
      temp[ichan] = sqrt(variance[ichan]);
    dump (stamp, ipol, nchan, data, ".bps");
    
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
