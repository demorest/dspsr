#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <cstdlib>
#include <string>
#include "machine_endian.h"
#include "wapp_head.h"

using namespace std;

int main(int argc, char **argv)
{
  struct WAPP_HEADER *wapphd=new WAPP_HEADER();
  int fd;
  fd=open(argv[1],O_RDONLY);
  readheader(fd,wapphd);
  cout << "headerversion=" << wapphd->header_version << endl;
  cout << "Source Name=" << wapphd->src_name << endl;
  cout << "Observation Type=" << wapphd->obs_type << endl;
  cout << "Observation Time=" << wapphd->obs_time << endl;
  cout << "Header Size=" << wapphd->header_size << endl;
  cout << "Observation date=" << wapphd->obs_date << endl;
  cout << "Start Time=" << wapphd->start_time << endl;
  cout << "Sample Time=" << wapphd->samp_time << endl;
  cout << "Wapp Time=" << wapphd->wapp_time << endl;
  cout << "Number of Lags=" << wapphd->num_lags << endl;
  cout << "Number of IFs=" << wapphd->nifs << endl;
  cout << "Level (1->3,2->9)=" << wapphd->level << endl;
  cout << "LagFormat=" << wapphd->lagformat << endl;
  cout << "Lagtrunc=" << wapphd->lagtrunc << endl;
  cout << "Central Observing Freq=" << wapphd->cent_freq << endl;
  cout << "Bandwidth=" << wapphd->bandwidth << endl;
  cout << "FreqInversion=" << wapphd->freqinversion << endl;
  cout << "Right Ascension=" << wapphd->src_ra << endl;
  cout << "Declination=" << wapphd->src_dec << endl;
  cout << "Start Azimuth=" << wapphd->start_az << endl;
  cout << "Start Zenith Angle=" << wapphd->start_za << endl;
  cout << "Start AST=" << wapphd->start_ast << endl;
  cout << "Start LST=" << wapphd->start_lst << endl;
  cout << "Sum=" << wapphd->sum << endl;
  cout << "Project ID=" << wapphd->project_id << endl;
  cout << "Observers=" << wapphd->observers << endl;
  cout << "Dispersion Measure=" << wapphd->psr_dm << endl;
  cout << "Dumptime=" << wapphd->dumptime << endl;
  cout << "Nbins=" << wapphd->nbins << endl;
  return EXIT_SUCCESS;
}
