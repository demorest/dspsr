//-*-C++-*-

#ifndef __Plotter_h
#define __Plotter_h

#include <string>
#include <vector>
#include <utility>

#include <cpgplot.h>

#include "environ.h"
#include "genutil.h"

#include "h_plotdata.h"
#include "plot_params.h"
#include "genutil.h"

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

    Timeseries* get_input()
      { return input.get(); }
    void set_input(Timeseries* _input)
      { input = _input; }

    void set_cpgopen_response(string response)
      { cpgopen_response = response; }
    string get_cpgopen_response()
      { return cpgopen_response; }

    void set_sample_start(int64 _sample_start)
      { sample_start = _sample_start; }
    int64 get_sample_start()
      { return sample_start; }

    void set_sample_finish(int64 _sample_finish)
      { sample_finish = _sample_finish; }
    int64 get_sample_finish()
      { return sample_finish; }

    void set_title(string title)
      { params.back().set_title(title); }
    string get_title()
      { return params.back().get_title(); }

    void set_xlabel(string xlabel)
      { params.back().set_xlabel(xlabel); }
    string get_xlabel()
      { return params.back().get_xlabel(); }

    void set_ylabel(string ylabel)
      { params.back().set_ylabel(ylabel); }
    string get_ylabel()
      { return params.back().get_ylabel(); }
    
    // A pair is a chan-pol pair
    bool get_plot_all_pairs()
      { return plot_all_pairs; }
    void set_plot_all_pairs(bool _plot_all_pairs)
      {  plot_all_pairs = _plot_all_pairs; }

    // To use pairs_to_plot you must explicity set_plot_all_pairs(false)
    void add_to_pairs_to_plot(pair<int,int> chanpol)
      { if( !is_one_of(chanpol,pairs_to_plot) ) pairs_to_plot.push_back(chanpol); }
    void remove_from_pairs_to_plot(pair<int,int> chanpol);

    static vector<int> good_colours;
    static vector<int> bad_colours;
    static vector<int> initial_good_colours();
    static vector<int> initial_bad_colours();

  protected:
    
    virtual bool open_plotwindow();
    // Called by plot() to set up the plotdatas
    virtual bool set_plotdatas();
    // Called by plot() to set xxmin, etc in params
    virtual bool set_params();
    
    virtual bool plot_background();
    virtual bool set_colour_index(int iplotdata);
    virtual bool user_input();
    virtual bool close_plotwindow();

    virtual bool highlight(float& x,float& y);

    char user_char;

    Reference::To<Timeseries> input;

    string cpgopen_response;  // default is 101/xs
    int dev;

    vector<plot_params> params;
    vector<plotdata> plotdatas;
    
    // A pair is a chan-pol pair
    // When plot_all_pairs=true, pairs_to_plot is ignored
    bool plot_all_pairs;
    // To use pairs_to_plot you must explicity set_plot_all_pairs(false)
    vector<pair<int,int> > pairs_to_plot;

    /* plotting window limits in units of samples */
    int64 sample_start;
    int64 sample_finish;
  };

}

#endif // !defined(__Input_h)
