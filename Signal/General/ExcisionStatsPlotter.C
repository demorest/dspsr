/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/ExcisionStatsPlotter.h"
#include "dsp/ExcisionUnpacker.h"

#include <iostream>
#include <algorithm>

#include <math.h>
#include <stdio.h>

#include <cpgplot.h>

using namespace std;

dsp::ExcisionStatsPlotter::ExcisionStatsPlotter ()
{
  theory_calculated = false;
  theory_max = 0.0;

  show_cutoff_sigma = true;
  plot_only_range = false;
  hist_min = 0.01;
}

dsp::ExcisionStatsPlotter::~ExcisionStatsPlotter ()
{
}

string dsp::ExcisionStatsPlotter::get_xlabel () const
{
  char label[64];
  snprintf (label, 64, "Low state count (in %d pts)",
            twobit->get_ndat_per_weight() * twobit->get_ndim_per_digitizer());
  return label;
}

string dsp::ExcisionStatsPlotter::get_ylabel () const
{
  return "Number of weights";
}


void dsp::ExcisionStatsPlotter::set_data (const HistUnpacker* hist)
{
  const ExcisionUnpacker* stats = dynamic_cast<const ExcisionUnpacker*>(hist);
  if (!stats)
    throw Error (InvalidParam, "dsp::ExcisionStatsPlotter::set_data",
                 "HistUnpacker is not a ExcisionUnpacker");

  twobit = stats;
  theory_calculated = false;

  BitStatsPlotter::set_data (hist);
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
float factln(int n)
{
  return gammln (n+1.0);
}

void dsp::ExcisionStatsPlotter::calculate_theory ()
{
  if (!twobit)
  {
    cerr << "ExcisionStatsPlotter::calculate_theory no data" << endl;
    return;
  }

  if (twobit->get_ndat_per_weight() < 1)
  {
    cerr << "ExcisionStatsPlotter::calculate_theory invalid data";
    return;
  }

  if (theory_calculated)
    return;

  // the number of samples per statistical measure
  int L = twobit->get_ndat_per_weight() * twobit->get_ndim_per_digitizer();

  theory.resize (L);
  theory_max = 0.0;

  float flnppwt = factln (L);

  double value = 0.0;
  double fraction_of_samples = 0.0;
  double fraction_ones = twobit->get_fraction_low();

  for (int wt=0; wt<L; wt++)
  {
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

double dsp::ExcisionStatsPlotter::get_chi_squared (int idig)
{
  if (!twobit)
    return 0;

  calculate_theory ();

  // get the histogram for this digitizer
  twobit->get_histogram (histogram, idig);

  // the theoretical binomial distribution must be scaled to the
  // number of weights tested
  double nweights = twobit->get_histogram_total (idig);

  unsigned ndim = twobit->get_ndim_per_digitizer ();

  double chisq = 0;
  for (unsigned iwt=0; iwt<histogram.size(); iwt++)
  {
    double normval = histogram[iwt] / nweights;
    double offmodel = normval - theory[iwt*ndim];
    chisq += offmodel * offmodel;
  }

  return chisq;
}



void dsp::ExcisionStatsPlotter::set_theory_colour ()
{
  theory_colour = 7; 

  for (int ic=0; ic<(int)colours.size(); ic++)
    if (colours[ic] == theory_colour)
    {
      theory_colour --;
      ic = -1;
    }
}


void dsp::ExcisionStatsPlotter::check_colours ()
{
  BitStatsPlotter::check_colours ();
  set_theory_colour();
}




bool dsp::ExcisionStatsPlotter::special (unsigned imin, unsigned imax,
				       float& ymax)
{  
  // calculate the theortical distribution of 1-count histogram
  calculate_theory ();
  
  // the theoretical binomial distribution must be scaled to the
  // number of weights tested
  double nweights = twobit->get_histogram_total (0);
  unsigned ndat_per_weight = twobit->get_ndat_per_weight ();

  unsigned ndim = twobit->get_ndim_per_digitizer ();
  xscale = ndim;

  vector<float> plot_theory (ndat_per_weight);
  for (unsigned iwt=0; iwt<ndat_per_weight; iwt++)
    plot_theory[iwt] = theory[iwt*ndim] * nweights * ndim;
  
  float hp_min, hp_max;
  unsigned n_min = twobit->get_nlow_min ();
  unsigned n_max = twobit->get_nlow_max ();

  if (plot_only_range)
  {
    hp_min = n_min - 10;
    hp_max = n_max + 10;
  }
  else
  {
    // definitely keep the theory in sight
    hp_min=0; hp_max=ndat_per_weight-1;
    for (; hp_min<ndat_per_weight; hp_min++)
      if (theory[unsigned(hp_min*ndim)] > theory_max*hist_min)
	break;
    for (; hp_max>0; hp_max--)
      if (theory[unsigned(hp_max*ndim)] > theory_max*hist_min)
	break;
  }

  // adjust to keep theoretical values in plot
  if (hp_min < imin)
    imin = (unsigned) hp_min;
  if (hp_max > imax)
    imax = (unsigned) hp_max;

  // want to keep the entire theoretical curve in the box
  if (ymax < theory_max * nweights * ndim)
    ymax = theory_max * nweights * ndim;

  ymax *= 1.05;

  // set the world coordinates for the histograms and draw a box
#ifdef DEBUG
  fprintf (stderr, "histogram (%d->%d) ymax: %f\n", imin, imax, ymax);
#endif

  cpgswin (imin*xscale, imax*xscale, ymax*hist_min, ymax);

  // plot the theoretical distribution of number of ones
  cpgsci(theory_colour);
  cpgpt (plot_theory, -1);

  // draw the cut-off sigma lines
  if (show_cutoff_sigma)
  {
    cpgmove (n_min, 0.0);
    cpgdraw (n_min, theory_max * nweights * ndim);
    cpgmove (n_max, 0.0);
    cpgdraw (n_max, theory_max * nweights * ndim);
  }

  return true;
}

