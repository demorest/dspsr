//-*-C++-*-

#ifndef __Plotter_h
#define __Plotter_h

#include <string>
#include <vector>
#include <utility>

#include <cpgplot.h>

#include "environ.h"
#include "genutil.h"
#include "MJD.h"

#include "PlotData.h"
#include "PlotParams.h"
#include "Timeseries.h"
#include "Ready.h"
#include "Reference.h"

namespace dsp {
  
  class Plotter: public Ready, public Reference::Able{

  public:

    Plotter();
    //virtual ~Plotter();

    virtual bool plot();

    virtual bool EnsureReadiness();

    Timeseries* get_input(){ return input.get(); }
    void set_input(Timeseries* _input){ input = _input; }

    void set_cpgopen_response(string response){ cpgopen_response = response; }
    string get_cpgopen_response(){ return cpgopen_response; }

    void set_sample_start(int64 _sample_start){ sample_start = _sample_start; }
    int64 get_sample_start(){ return sample_start; }

    void set_sample_finish(int64 _sample_finish){ sample_finish = _sample_finish; }
    int64 get_sample_finish(){ return sample_finish; }

    void set_base_title(string _base_title){base_title = _base_title; }
    string get_base_title(){ return base_title; }

    void set_xlabel(string _xlabel){ xlabel = _xlabel; }
    string get_xlabel(){ return xlabel; }

    void set_ylabel(string _ylabel){ ylabel = _ylabel; }    
    string get_ylabel(){ return ylabel; }

    void set_referenceMJD(const MJD& ref){ referenceMJD = ref; }
    const MJD& get_referenceMJD(){ return referenceMJD; } 

    // A pair is a chan-pol pair
    bool get_plot_all_pairs(){ return plot_all_pairs; }
    void set_plot_all_pairs(bool _plot_all_pairs){  plot_all_pairs = _plot_all_pairs; }

    // To use pairs_to_plot you must explicity set_plot_all_pairs(false)
    void add_to_pairs_to_plot(pair<int,int> chanpol)
      { if( !is_one_of(chanpol,pairs_to_plot) ) pairs_to_plot.push_back(chanpol); }
    void remove_from_pairs_to_plot(pair<int,int> chanpol);

    char get_user_char(){ return user_char; }

  protected:
    
    virtual bool open_plotwindow();
    virtual bool close_plotwindow();

    // Called by plot() to set up the plotdatas
    virtual bool set_plotdatas();
    // Called by plot() to set xxmin, etc in params
    virtual bool set_params();
    
    virtual bool plot_background();
    virtual bool set_colour_index(int iplotdata);
    virtual bool user_input();
    
    virtual bool highlight(float& x,float& y);

    char user_char;

    Reference::To<Timeseries> input;

    string cpgopen_response;  // default is 101/xs
    int dev;

    vector<plot::PlotParams> params;
    vector<plot::PlotData> plotdatas;
    
    string base_title;
    string xlabel;
    string ylabel;

    // A pair is a chan-pol pair
    // When plot_all_pairs=true, pairs_to_plot is ignored
    bool plot_all_pairs;
    // To use pairs_to_plot you must explicity set_plot_all_pairs(false)
    vector<pair<int,int> > pairs_to_plot;

    // Used for setting xaxis scale (time)
    MJD referenceMJD;

    /* plotting window limits in units of samples */
    int64 sample_start;
    int64 sample_finish;
  };

}

#endif // !defined(__Input_h)
