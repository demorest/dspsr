#include <iostream>
#include <algorithm>

#include <math.h>
#include <stdio.h>

#include <cpgplot.h>

#include "dsp/TwoBitStatsPlotter.h"
#include "dsp/TwoBitCorrection.h"
#include "genutil.h"

dsp::TwoBitStatsPlotter::TwoBitStatsPlotter ()
{
  theory_calculated = false;
  theory_max = 0.0;
  data = 0;

  show_cutoff_sigma = true;
  plot_only_range = false;
  horizontal = true;

  full_xscale = false;
  hist_min = 0.01;

  vpxmin = 0.1;
  vpxmax = 0.9;
  vpymin = 0.1;
  vpymax = 0.9;
}

dsp::TwoBitStatsPlotter::~TwoBitStatsPlotter ()
{
}

void dsp::TwoBitStatsPlotter::set_data (const TwoBitCorrection* stats)
{
  data = stats;
  theory_calculated = false;
}

float gammln(float xx)
{
  /* Numerical Recipes */
  static double cof [6]= { 76.18009172947146, -86.50532032941677,
			   24.01409824083091, -1.231739572450155,
			   0.1208650973866179e-2, -0.5395239384953e-5 };
  double x, y, tmp, ser;
  int j;

  y = x = xx;
  tmp = x + 5.5;
  tmp -= (x+0.5) * log(tmp);
  ser = 1.000000000190015;
  for (j=0; j<=5; j++) {
    y++;
    ser += cof[j]/y;
  }
  return -tmp + log(2.5066282746310005 * ser/x);
}

// returns ln(n!)
float factln(int n) {
  return gammln (n+1.0);
}

void dsp::TwoBitStatsPlotter::calculate_theory ()
{
  if (!data) {
    cerr << "TwoBitStatsPlotter::calculate_theory no data" << endl;
    return;
  }

  if (data->get_nsample() < 1) {
    cerr << "TwoBitStatsPlotter::calculate_theory invalid data";
    return;
  }

  if (theory_calculated)
    return;

  // the number of samples per statistical measure
  int L = data->get_nsample();

  theory.resize (L);
  theory_max = 0.0;

  float flnppwt = factln (L);

  double value = 0.0;
  double fraction_of_samples = 0.0;
  double fraction_ones = data->get_fraction_low();

  for (int wt=0; wt<L; wt++) {

    fraction_of_samples = double(wt) / double (L);

    value = exp(flnppwt - factln(wt) - factln(L-wt) +
		L * (log(pow (fraction_ones, fraction_of_samples)) +
		     log(pow (1.0-fraction_ones, 1.0-fraction_of_samples))));
    
    if (value > theory_max) 
      theory_max = value;

    theory[wt] = value;

  }

  theory_calculated = true;
}

double dsp::TwoBitStatsPlotter::get_chi_squared (int idig)
{
  if (!data)
    return 0;

  calculate_theory ();

  // get the histogram for this digitizer
  data->get_histogram (histogram, idig);

  // the theoretical binomial distribution must be scaled to the
  // number of weights tested
  double nweights = data->get_histogram_total (idig);

  double chisq = 0;
  for (unsigned iwt=0; iwt<histogram.size(); iwt++) {
    double normval = histogram[iwt] / nweights;
    double offmodel = normval - theory[iwt];
    chisq += offmodel * offmodel;
  }

  return chisq;
}

// helper function
void cpgpt (vector<float>& vals, int type)
{
  for (unsigned i=0; i<vals.size(); i++) {
    float ind = i;
    float val = vals[i];
    cpgpt (1, &ind, &val, type);
  }
}

void dsp::TwoBitStatsPlotter::set_theory_colour ()
{
  theory_colour = 7; 

  for (int ic=0; ic<(int)colours.size(); ic++)
    if (colours[ic] == theory_colour) {
      theory_colour --;
      ic = -1;
    }
}

void dsp::TwoBitStatsPlotter::set_colours (const vector<int>& ucolours) 
{
  colours = ucolours;
  check_colours();
  set_theory_colour();
}

void dsp::TwoBitStatsPlotter::check_colours ()
{
  if (!data)
    return;

  int defc[2] = { 10, 13 };

  int ndig = data->get_ndig ();
  if (ndig < 0)
    return;

  if (colours.size() < (unsigned)ndig) {
    int available = colours.size();
    colours.resize(ndig);

    if (available == 0)
      for (unsigned ic=0; ic<colours.size(); ic++)
	colours[ic] = defc[ic%2];

    else
      for (unsigned ic=available; ic<colours.size(); ic++)
	colours[ic] = colours[ic%available];

    set_theory_colour();
  }
}

void dsp::TwoBitStatsPlotter::pglabel()
{
  if (!data)
    return;

  check_colours ();
  cpgsci (1);
  cpgsch (0.75);

  int nsample = data->get_nsample();
  if (nsample < 0)
    return;

  int ndig = data->get_ndig ();
  if (ndig < 0)
    return;

  char label [80];
  sprintf (label, "Number of ones (in %d pts)", nsample );
  cpglab (label, "Number of weights", " ");

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
  // not necessary?
  //  cpgsci (theory_colour);
  //  cpgmtxt("T", .5, 0.5, 0.5, "theory");
}

void dsp::TwoBitStatsPlotter::set_viewport (float _vpxmin, float _vpxmax,
					    float _vpymin, float _vpymax)
{
  vpxmin = _vpxmin;
  vpxmax = _vpxmax;
  vpymin = _vpymin;
  vpymax = _vpymax;
}

void dsp::TwoBitStatsPlotter::plot ()
{
  if (!data)
    return;

  check_colours ();

  int nsample = data->get_nsample();
  if (nsample < 0)
    return;

  int ndig = data->get_ndig ();
  if (ndig < 0)
    return;

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

void dsp::TwoBitStatsPlotter::pgplot (int poln)
{
  if (!data)
    return;

  check_colours ();

  int nsample = data->get_nsample();
  if (nsample < 0)
    return;

  int ndig = data->get_ndig ();
  if (ndig < 0)
    return;
  
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
      throw_str ("TwoBitStatsPlotter::pgplot: empty range idig=%d", idig);
    
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
  
  // calculate the theortical distribution of 1-count histogram
  calculate_theory ();
  
  // the theoretical binomial distribution must be scaled to the
  // number of weights tested
  double nweights = data->get_histogram_total (0);
  
  vector<float> plot_theory (nsample);
  for (int iwt=0; iwt<nsample; iwt++)
    plot_theory[iwt] = theory[iwt] * nweights;
  
  float hp_min, hp_max;
  int n_min = data->get_nmin ();
  int n_max = data->get_nmax ();

  if (plot_only_range)  {
    hp_min = n_min - 10;
    hp_max = n_max + 10;
  }
  else  {
    // definitely keep the theory in sight
    hp_min=0; hp_max=nsample-1;
    for (; hp_min<nsample; hp_min++)
      if (theory[hp_min] > theory_max*hist_min)
	break;
    for (; hp_max>0; hp_max--)
      if (theory[hp_max] > theory_max*hist_min)
	break;
  }

  // adjust to keep theoretical values in plot
  if (hp_min < imin)
    imin = (int) hp_min;
  if (hp_max > imax)
    imax = (int) hp_max;

  // want to keep the entire theoretical curve in the box
  if (ymax < theory_max * nweights)
    ymax = theory_max * nweights;

  ymax *= 1.05;

  // set the world coordinates for the histograms and draw a box
#ifdef DEBUG
  fprintf (stderr, "histogram (%d->%d) ymax: %f\n", imin, imax, ymax);
#endif

  cpgsch(0.5);
  cpgswin (imin, imax, ymax*hist_min, ymax);
  cpgsci (1);
  cpgsls (1);
  cpgbox  ("bcnst",0.0,0,"bcnvst",0.0,0);

  // plot the theoretical distribution of number of ones
  cpgsci(theory_colour);
  cpgpt (plot_theory, -1);

  // draw the cut-off sigma lines
  if (show_cutoff_sigma) {
    cpgmove (n_min, 0.0);
    cpgdraw (n_min, theory_max * nweights);
    cpgmove (n_max, 0.0);
    cpgdraw (n_max, theory_max * nweights);
  }

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
