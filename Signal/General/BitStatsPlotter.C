/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitStatsPlotter.h"
#include "dsp/HistUnpacker.h"

#include <cpgplot.h>

#include <algorithm>

using namespace std;

dsp::BitStatsPlotter::BitStatsPlotter ()
{
  horizontal = true;
  full_xscale = false;
  hist_min = 0.00;
  xscale = 1.0;

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
  for (unsigned i=0; i<vals.size(); i++)
  {
    float ind = i * xscale;
    float val = vals[i];
    if (val)
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

void dsp::BitStatsPlotter::label()
{
  if (!data)
    return;

  check_colours ();
  cpgsci (1);
  cpgsch (0.8);

  string xlabel = get_xlabel();
  string ylabel = get_ylabel();

  cpglab (xlabel.c_str(), ylabel.c_str(), " ");

  if (data->has_input()
      && data->get_input()->get_state()==Signal::Analytic
      && data->get_ndim_per_digitizer() == 1)
  {
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
  label ();

  cpgsvp (vpxmin, x1, y2, vpymax);
  plot (0,0);

  cpgsvp (x2, vpxmax, vpymin, y1);
  plot (0,1);

}



void dsp::BitStatsPlotter::plot (unsigned ichan, unsigned ipol)
{
  if (!data)
    return;

#ifdef _DEBUG
  cerr << "dsp::BitStatsPlotter::plot"
    " ichan=" << ichan << " ipol=" << ipol << endl;
#endif

  check_colours ();

  unsigned nstate = data->get_nstate();
  unsigned ndig = data->get_ndig ();

#ifdef _DEBUG
  cerr << "dsp::BitStatsPlotter::plot"
    " nstate=" << nstate << " ndig=" << ndig << endl;
#endif
 
  unsigned nplot = 1;

  // plot histograms for in-phase and quadrature components
  if (data->has_input() 
      && data->get_input()->get_state() == Signal::Analytic
      && data->get_ndim_per_digitizer() == 1)
    nplot = 2;

  int idig[2] = {-1, -1};

  float ymax = 0.0;

  int imin = nstate-1;
  int imax = 0;

  unsigned iplot=0; 

  for (iplot=0; iplot < nplot; iplot++) {

    // find the digitizer channel
    for (unsigned jdig=0; jdig < ndig; jdig++) {

      if (data->get_output_ichan(jdig)  == ichan &&
	  data->get_output_ipol(jdig)   == ipol &&
	  data->get_output_offset(jdig) == iplot)
	{
	  if (idig[iplot] != -1)
	    throw Error (InvalidState, "dsp::BitStatsPlotter::plot",
			 "Both digitizer channels %d and %d match "
			 "ichan=%d ipol=%d ioff=%d",
			 idig[iplot], jdig, ichan, ipol, iplot);
	  idig[iplot] = jdig;
	}
    }

    if (idig[iplot] == -1)
      throw Error (InvalidState, "dsp::BitStatsPlotter::plot",
		   "No digitizer channel matches "
		   "ichan=%d ipol=%d ioff=%d",
		   ichan, ipol, iplot);

    // find the range of the data to be plotted
    vector<float>::iterator maxel;

#ifdef _DEBUG
    cerr << "dsp::BitStatsPlotter::plot idig=" << idig[iplot] << endl;
#endif

    data->get_histogram (histogram, idig[iplot]);

    maxel = max_element(histogram.begin(), histogram.end());
    
    if (maxel == histogram.end())
      throw Error (InvalidParam, "BitStatsPlotter::plot",
		   "empty range idig=%d", idig[iplot]);
    
    ymax = max (ymax, *maxel);
    
    if (!full_xscale)
    {
      int imin_orig = imin;
      for (imin=0; imin<imin_orig; imin++)
	if (histogram[imin] > ymax * hist_min)
	  break;
      int imax_orig = imax;
      for (imax=nstate-1; imax>imax_orig; imax--)
	if (histogram[imax] > ymax * hist_min)
	  break;
    }
  }

  // set the world coordinates for the histograms and draw a box
  if (!special (imin, imax, ymax))
  {
    ymax *= 1.05;
    cpgswin (imin, imax, ymax*hist_min, ymax);
  }

  cpgsci (1);
  cpgsls (1);
  cpgsch (0.8);
  cpgbox  ("bcnst",0.0,0,"bcnvst",0.0,0);

  char label [64];
  sprintf (label, "Chan:%d - Poln:%d", ichan, ipol);

  cpgmtxt("T", -1.5, 0.05, 0.0, label);

  // plot the actual distribution of number of ones
  float midheight = ymax/2.0;

  for (iplot=0; iplot < nplot; iplot++)
  {
    data->get_histogram (histogram, idig[iplot]);

    cpgsci (colours[iplot]);
    cpgpt (histogram, 2);
    
    float fractone = data->get_histogram_mean (idig[iplot]);
    cpgsls (4);
    cpgmove (fractone*float(nstate), 0.0);
    cpgdraw (fractone*float(nstate), midheight);
  }

  cpgsls (1);
}

