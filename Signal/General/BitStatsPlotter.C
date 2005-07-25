#include "dsp/BitStatsPlotter.h"
#include "dsp/HistUnpacker.h"

#include <cpgplot.h>


dsp::BitStatsPlotter::BitStatsPlotter ()
{
  horizontal = true;
  full_xscale = false;
  hist_min = 0.00;

  vpxmin = 0.1;
  vpxmax = 0.9;
  vpymin = 0.1;
  vpymax = 0.9;
}

dsp::BitStatsPlotter::~BitStatsPlotter ()
{
}

string dsp::BitStatsPlotter::get_xlabel () const
{
  return "Sample value";
}

string dsp::BitStatsPlotter::get_ylabel () const
{
  return "Number of weights";
}

void dsp::BitStatsPlotter::set_data (const HistUnpacker* stats)
{
  data = stats;
}



// helper function
void dsp::BitStatsPlotter::cpgpt (std::vector<float>& vals, int type)
{
  for (unsigned i=0; i<vals.size(); i++) {
    float ind = i;
    float val = vals[i];
    ::cpgpt (1, &ind, &val, type);
  }
}

void dsp::BitStatsPlotter::set_colours (const vector<int>& ucolours) 
{
  colours = ucolours;
  check_colours();
}

void dsp::BitStatsPlotter::check_colours ()
{
  if (!data)
    return;

  int defc[2] = { 10, 13 };

  unsigned ndig = data->get_ndig ();

  if (colours.size() < ndig) {
    int available = colours.size();
    colours.resize(ndig);

    if (available == 0)
      for (unsigned ic=0; ic<colours.size(); ic++)
	colours[ic] = defc[ic%2];

    else
      for (unsigned ic=available; ic<colours.size(); ic++)
	colours[ic] = colours[ic%available];

  }
}

void dsp::BitStatsPlotter::pglabel()
{
  if (!data)
    return;

  check_colours ();
  cpgsci (1);
  cpgsch (0.75);

  unsigned ndig = data->get_ndig ();

  string xlabel = get_xlabel();
  string ylabel = get_ylabel();

  cpglab (xlabel.c_str(), ylabel.c_str(), " ");

  if (ndig == 4) {
    cpgsci (colours[0]);
    cpgmtxt("T", .5, 0.0, 0.0, "In-phase");
    cpgsci (colours[1]);
    cpgmtxt("T", .5, 1.0, 1.0, "Quadrature");
    if (colours[0] != colours[2]) {
      cpgsci (1);
      cpgmtxt("T", .5, 0.5, 0.5, "Left");
      cpgmtxt("T", 1.5, 0.5, 0.5, "Right");

      cpgsci (colours[2]);
      cpgmtxt("T", 1.5, 0.0, 0.0, "In-phase");
      cpgsci (colours[3]);
      cpgmtxt("T", 1.5, 1.0, 1.0, "Quadrature");
    }
  }

}

void dsp::BitStatsPlotter::set_viewport (float _vpxmin, float _vpxmax,
					 float _vpymin, float _vpymax)
{
  vpxmin = _vpxmin;
  vpxmax = _vpxmax;
  vpymin = _vpymin;
  vpymax = _vpymax;
}

void dsp::BitStatsPlotter::plot ()
{
  if (!data)
    return;

  check_colours ();

  unsigned ndig = data->get_ndig ();

  float x1=vpxmax;
  float x2=vpxmin;
  float y1=vpymax;
  float y2=vpymin;

  float adjust;

  if (ndig > 1)
    if (horizontal)
      x1 = x2 = (vpxmax + vpxmin) / 2.0;
    else {
      y1 = y2 = (vpymax + vpymin) / 2.0;
      adjust = (vpymax - vpymin) * .025;
      y1 -= adjust;
      y2 += adjust;
    }

  // plot the labels
  cpgsvp (vpxmin, vpxmax, vpymin, vpymax);
  pglabel ();
  
  cpgsvp (vpxmin, x1, y2, vpymax);
  pgplot (0);
  cpgsci (1);
  cpgsch (0.8);
  cpgmtxt("T", -1.5, 0.05, 0.0, "Left Polarization");
  
  cpgsvp (x2, vpxmax, vpymin, y1);
  pgplot (1);
  cpgsci (1);
  cpgsch (0.8);
  cpgmtxt("T", -1.5, 0.05, 0.0, "Right Polarization");
}

void dsp::BitStatsPlotter::pgplot (unsigned poln)
{
  if (!data)
    return;

  check_colours ();

  unsigned nsample = data->get_nsample();
  unsigned ndig = data->get_ndig ();
  
  int istat = 0;
  int iendt = 0;

  if (poln == 4)
    iendt = ndig -1;
  else {
    istat = poln * (ndig / 2);
    if (ndig == 4)
      iendt = istat + 1;
    else
      iendt = istat;
  }

  // find the range of the data to be plotted
  vector<float>::iterator maxel;
  float ymax = 0.0;
  int imax=0, imin=nsample-1;

  int idig = 0;
  for (idig=istat; idig <= iendt; idig++) {

    data->get_histogram (histogram, idig);

    maxel = max_element(histogram.begin(), histogram.end());
    
    if (maxel == histogram.end())
      throw_str ("BitStatsPlotter::pgplot: empty range idig=%d", idig);
    
    ymax = max (ymax, *maxel);
    
    if (!full_xscale) {
      int imin_orig = imin;
      for (imin=0; imin<imin_orig; imin++)
	if (histogram[imin] > ymax * hist_min)
	  break;
      int imax_orig = imax;
      for (imax=nsample-1; imax>imax_orig; imax--)
	if (histogram[imax] > ymax * hist_min)
	  break;
    }
  }

  // set the world coordinates for the histograms and draw a box
#ifdef DEBUG
  fprintf (stderr, "histogram (%d->%d) ymax: %f\n", imin, imax, ymax);
#endif

  cpgsch(0.5);

  if (!special (imin, imax, ymax)) {
    ymax *= 1.05;
    cpgswin (imin, imax, ymax*hist_min, ymax);
  }

  cpgsci (1);
  cpgsls (1);
  cpgbox  ("bcnst",0.0,0,"bcnvst",0.0,0);

  // plot the actual distribution of number of ones
  float midheight = ymax/2.0;

  for (idig=istat; idig <= iendt; idig++) {

    data->get_histogram (histogram, idig);

    cpgsci (colours[idig]);
    cpgpt (histogram, 2);
    
    float fractone = data->get_histogram_mean (idig);
    cpgsls (4);
    cpgmove (fractone*float(nsample), 0.0);
    cpgdraw (fractone*float(nsample), midheight);

  }

  cpgsls (1);
}
