//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Attic/TwoBitStatsPlotter.h,v $
   $Revision: 1.1 $
   $Date: 2002/07/12 14:15:23 $
   $Author: wvanstra $ */

#ifndef __TwoBitStatsPlotter_h
#define __TwoBitStatsPlotter_h

#include <vector>

namespace dsp {
  
  class TwoBitCorrection;

  //! Plots the histograms maintained by TwoBitCorrection
  class TwoBitStatsPlotter {

  public:

    //! Null constructor
    TwoBitStatsPlotter () { init(); }

    //! Virtual destructor
    virtual ~TwoBitStatsPlotter () { }

    //! Set the data to be plotted
    void set_data (const TwoBitCorrection* stats);

    //! Plot the data in the currently open viewport
    void plot ();

    //! Get a measure of the difference between the histogram and the theory
    double get_chi_squared (int chan);

    //! Plot vertical bars to indicate the cut-off thresholds
    bool show_cutoff_sigma;

    //! Plot the two polarizations beside eachother
    bool horizontal;

    //! Plot the entire x-axis of the histogram
    bool full_xscale;

    //! Fraction of histogram maximum below which points are ignored
    float hist_min;

  protected:

    //! Where individual histograms are kept for plotting
    vector<float> histogram;

    //! Theoretical, optimal histogram
    vector<float> theory;

    //! Maxmimum value of theory
    float theory_max;

    //! Set true when the theoretical, optimal histogram is calculated
    bool theory_calculated;

    //! Colour used when plotting theoretical
    int theory_colour;

    //! Colour used when plotting histogram from each channel
    vector <int> colours;

    //! Data to be plotted
    const TwoBitCorrection* data;

    //!
    void calculate_theory ();

    //! Label the plot
    void pglabel ();

    void pgplot (int poln);

    void pgplots (float vpxmin, float vpxmax, float vpymin, float vpymax);

    // allow the user to modify the colour used for each channel
    void set_colours (const vector<int>& colours);

    int  get_theory_colour () { return theory_colour; };

    void set_theory_colour ();

    void check_colours ();

    //! Initialize null values
    void init();
  };
  
}

#endif
