//-*-C++-*-

#ifndef __Plotter_h
#define __Plotter_h

#include <string>

#include "BasicPlotter.h"

namespace dsp {
  
  class Plotter: public BasicPlotter{

  public:

    Plotter();

    virtual bool plot();
    virtual void close();

    void set_cpgopen_response(string response){ cpgopen_response = response; }
    string get_cpgopen_response(){ return cpgopen_response; }

    void set_base_title(string _base_title){base_title = _base_title; }
    string get_base_title(){ return base_title; }

    void set_xlabel(string _xlabel){ xlabel = _xlabel; }
    string get_xlabel(){ return xlabel; }

    void set_ylabel(string _ylabel){ ylabel = _ylabel; }    
    string get_ylabel(){ return ylabel; }

  protected:
    
    virtual bool open_plotwindow();
    
    virtual bool plot_background();
    virtual bool user_input();
    
    virtual bool highlight(float& x,float& y);

    string cpgopen_response;  // default is 101/xs
    int dev;

    string base_title;
    string xlabel;
    string ylabel;

  };

}

#endif // !defined(__Plotter_h)
