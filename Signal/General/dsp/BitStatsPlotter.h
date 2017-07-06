//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/BitStatsPlotter.h

#ifndef __BitStatsPlotter_h
#define __BitStatsPlotter_h

#include <vector>

#include "Reference.h"

namespace dsp {
  
  class HistUnpacker;

  //! Plots the histograms maintained by HistUnpacker classes
  class BitStatsPlotter : public Reference::Able {

  public:

    //! Null constructor
    BitStatsPlotter ();

    //! Virtual destructor
    virtual ~BitStatsPlotter ();

    //! Set the data to be plotted
    virtual void set_data (const HistUnpacker* stats);

    //! Set the plot colour used for each digitizer
    void set_colours (const std::vector<int>& colours);

    //! Plot the histogram for the specified frequency channel and polarization
    void plot (unsigned ichan, unsigned ipol);

    //! Label the plot
    void label ();

    //! The label on the x-axis
    virtual std::string get_xlabel () const;

    //! The label on the y-axis
    virtual std::string get_ylabel () const;

    //! Plot the entire x-axis of the histogram
    bool full_xscale;

    //! Fraction of histogram maximum below which points are ignored
    float hist_min;

    /** @name Backward Compatibility
     *  These methods remain only to provide backward compatibility
     *  with older software.
     */
    //@{

    //! Set the device-normalized coordinates of viewport frame
    void set_viewport (float xmin, float xmax, float ymin, float ymax);

    //! Plot the two polarizations beside eachother
    bool horizontal;

    //! Plot two polarizations in the current viewport
    void plot ();

    //@}

  protected:

    //! Any special plotting implemented in derived classes
    /*! Implementations should return true if cpgswin is called */
    virtual bool special (unsigned imin, unsigned imax, float& ymax)
	  { return false; }

    //! Where individual histograms are kept for plotting
    std::vector<float> histogram;

    //! Colour used when plotting histogram from each digitizer
    std::vector<int> colours;

    //! Ensure that colours for different digitizers are unique
    virtual void check_colours ();

    //! Plot a vector of points with index on x-axis
    void cpgpt (std::vector<float>& vals, int type);

    //! Scale factor multplied by each index on x axis
    float xscale;

  private:

   //! Device normalized coordinates of viewport frame
    float vpxmin;
    float vpxmax;
    float vpymin;
    float vpymax;

    //! Data to be plotted
    Reference::To<const HistUnpacker> data;


  };
  
}

#endif
