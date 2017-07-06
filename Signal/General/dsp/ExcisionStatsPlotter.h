//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/ExcisionStatsPlotter.h

#ifndef __ExcisionStatsPlotter_h
#define __ExcisionStatsPlotter_h

#include "dsp/BitStatsPlotter.h"

namespace dsp {
  
  class ExcisionUnpacker;

  //! Plots the histograms maintained by ExcisionUnpacker
  class ExcisionStatsPlotter : public BitStatsPlotter {

  public:

    //! Null constructor
    ExcisionStatsPlotter ();

    //! Virtual destructor
    virtual ~ExcisionStatsPlotter ();

    //! Set the data to be plotted
    void set_data (const HistUnpacker* stats);

    //! The label on the x-axis
    std::string get_xlabel () const;

    //! The label on the y-axis
    std::string get_ylabel () const;

     //! Get the colour used to plot the theoretical distribution
    int get_theory_colour () { return theory_colour; };

    //! Get a measure of the difference between the histogram and the theory
    double get_chi_squared (int idig);

    //! Plot vertical bars to indicate the cut-off thresholds
    bool show_cutoff_sigma;

    //! Plot vertical bars to indicate the cut-off thresholds
    bool plot_only_range;

  protected:

    //! Theoretical, optimal histogram
    std::vector<float> theory;

    //! Maxmimum value of theory
    float theory_max;

    //! Set true when the theoretical, optimal histogram is calculated
    bool theory_calculated;

    //! Colour used when plotting theoretical
    int theory_colour;

    //! Data to be plotted
    Reference::To<const ExcisionUnpacker> twobit;

    void calculate_theory ();
    void set_theory_colour ();
    void check_colours ();
    bool special (unsigned imin, unsigned imax, float& ymax);

  };
  
}

#endif
