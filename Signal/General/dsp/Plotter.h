//-*-C++-*-

#ifndef __Plotter_h
#define __Plotter_h

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <cpgplot.h>
#include <stdlib.h>

#include "environ.h"
#include "genutil.h"
#include "string_utils.h"
#include "minmax.h"
#include "MJD.h"

#include "PlotParams.h"
#include "Ready.h"

#include "dsp/BasicPlotter.h"

namespace dsp {

  template<class DataType>
  class Plotter: public BasicPlotter<DataType>{

  public:

    Plotter();

    virtual ~Plotter();

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

template<class DataType>
dsp::Plotter<DataType>::Plotter() : BasicPlotter<DataType>() {
  name = "Plotter";

  cpgopen_response = "101/xs";
  dev = -1;
  
  base_title = " ";
  xlabel = "Time (s)";
  ylabel = "Intensity";

  mutated = true;
}

template<class DataType>
dsp::Plotter<DataType>::~Plotter(){ }

template<class DataType>
bool dsp::Plotter<DataType>::open_plotwindow(){
  dev = cpgopen(cpgopen_response.c_str());

  if(dev <=0 ){
    fprintf(stderr,"dsp::Plotter::open_plotwindow() cpgopen call failed\n");
    return false;
  }
  
  return true;
}

template<class DataType>
bool dsp::Plotter<DataType>::plot(){
  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"In dsp::Plotter::plot()\n");

  if( !EnsureReadiness() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to ensure readiness\n");
    return false;
  }

  if( !open_plotwindow() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to open plot window\n");
    return false;
  }

  if( !set_plotdatas() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to set plotdatas\n");
    return false;
  }
  if( plotdatas.size()==0 ){
    fprintf(stderr,"Error!  No plotdatas specified\n");
    return false;
  }
  if( !set_params() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to set params\n");
    return false;
  }  

  user_char = '\0';
  while( !(user_char=='q' || user_char=='Q') ){
    fprintf(stderr,"\nContinued in loop as user_char='%c'\n",user_char);
    
    if( mutated ){
      plot_background();
      for( unsigned iplotdata=0; iplotdata<plotdatas.size(); iplotdata++){
	set_colour_index(iplotdata);
	plotdatas[iplotdata]->plot(params.back());
      }
    }

    cpgsci(1);
    cpgsls(1);
    cpgsch(1.0);
    user_input();
  }

  close();

  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"Returning successfully from dsp::Plotter::plot()\n");
  return true;
}

template<class DataType>
void dsp::Plotter<DataType>::close(){
  cpgslct(dev);
  cpgclos();
}

template<class DataType>
bool dsp::Plotter<DataType>::user_input(){
  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"In dsp::Plotter::user_input()\n");
  
  float x=1.0;      // Used to store x co-ord of cursor upon keypress
  float y=1.0;      // Used to store y co-ord of cursor upon keypress
  float xref;       // Used to store secondary x co-ord of cursor upon e.g. completion of zoom box
  float yref;       // Used to store secondary y co-ord of cursor upon e.g. completion of zoom box
 
  if(cpgcurs(&x,&y,&user_char)==0){
    cerr<<"Error in cpgcurs\n";
    return false;
  }

  if(Ready::verbose || Operation::verbose)
    cerr<<"After cpgcurs user_char is '"<<user_char<<"'\n\n";

  if(user_char=='z'){
    if(Ready::verbose || Operation::verbose)
	cerr<<"Detected a 'z' at x='"<<x<<"' and y='"<<y<<"'\n";
    xref=x;
    yref=y;
    cpgband(2, 0, xref, yref, &x, &y, &user_char);
      
    if(x<xref){ float tmp=x; x=xref; xref=tmp; }
    if(y<yref){ float tmp=y; y=yref; yref=tmp; }

    params.push_back( plot::PlotParams(xref, x, yref, y, params.back()) );

    mutated = true;
    return true;
  }

  if(user_char=='n' || user_char=='d' ){
    highlight(x,y);
    params.push_back( params.back() );
    params.back().incr();
    
    mutated = true;
    return true;
  }
  
  if(user_char=='u' && params.size()>1){
    params.pop_back();

    mutated = true;
    return true;
  }
    
  if(user_char=='r'){
    params.push_back(plot::PlotParams(x,
				x+params.back().get_xmax()-params.back().get_xmin(),
				params.back().get_ymin(),
				params.back().get_ymax(),
				params.back()));
    mutated = true;
    return true;
  }
    
  if(user_char=='l'){
    params.push_back(plot::PlotParams(x-(params.back().get_xmax()-params.back().get_xmin()),
				x,
				params.back().get_ymin(),
				params.back().get_ymax(),
				params.back()));
    mutated = true;
    return true;
  }

  if(user_char=='o'){
    params.erase(params.begin()+1,params.end());
    mutated = true;
    return true;
  }

  if(user_char=='b'){
    params.back().switch_block_new_data();
    mutated = false;
    return true;
  }

  if(user_char=='c'){
    fprintf(stderr,"x=%f\ty=%f\n",x,y);
    mutated = false;
    return true;
  }

  if(user_char=='p'){
    string response = string("dsp_Plotter") + string(".cps/cps");;
    
    fprintf(stderr,"Will ps to '%s'\n",response.c_str());
    
    int psdev = cpgopen(response.c_str());

    if( psdev <= 0 ){
      fprintf(stderr,"dsp::Plotter::open_plotwindow() cpgopen call failed\n");
      return false;
    }
  
    plot_background();
    for( unsigned iplotdata=0; iplotdata<plotdatas.size(); iplotdata++){
      set_colour_index(iplotdata);
      plotdatas[iplotdata]->plot(params.back());
    }
   
    cpgclos();
    cpgslct(dev);

    fprintf(stderr,"Out of replot for ps file\n");

    mutated = false;
    return true;
  }

  if( user_char=='q' || user_char=='Q' ){
    return true;
  }

  fprintf(stderr,"You typed '%c'\n",user_char);
  mutated = false;

  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"Returning at end of dsp::Plotter::user_input()\n");

  return true;
}      

template<class DataType>
bool dsp::Plotter<DataType>::plot_background(){
  cpgeras();
  
  cpgsubp(1,1);  
  cpgask (0);                /*asks for user to hit return*/
  cpgsch (0.8);              /*set character height*/
  cpgsci (1);                /*set colour index*/
    
  cpgpanl(1, 1);  
  
  plot::PlotParams* last_params = &params.back();

  // Draw lower 'reminder' boxes
  cpgsvp (0.0, 1.0, 0.1, 0.2);
  cpgswin(0.0, 10.0, 0.0, 1.0);
  cpgbox("BCT",1.0,0,"BCT",1.0,0);
  
  cpgmtxt("B",-1.5,0.05,0.5,"Zoom [z]");
  cpgmove(1.0,0.0);  
  cpgdraw(1.0,1.0);
  cpgmtxt("B",-1.5,0.15,0.5,"Unzoom [u]");
  cpgmove(2.0,0.0);  
  cpgdraw(2.0,1.0);
  cpgmtxt("B",-3.0,0.25,0.5,"Go left[l]");
  cpgmtxt("B",-1.5,0.25,0.5,"Go right [r]");
  cpgmove(3.0,0.0);  
  cpgdraw(3.0,1.0);
  cpgmtxt("B",-1.5,0.35,0.5,"Write ps [p]");
  cpgmove(4.0,0.0);  
  cpgdraw(4.0,1.0);
  cpgmtxt("B",-3.0,0.45,0.5,"Start");
  cpgmtxt("B",-1.5,0.45,0.5,"over [o]");
  cpgmove(5.0,0.0);  
  cpgdraw(5.0,1.0);
  if(last_params->get_block_new_data()==false)
    cpgmtxt("B",-3.0,0.55,0.5,"Block new");
  else
    cpgmtxt("B",-3.0,0.55,0.5,"Allow new");
  cpgmtxt("B",-1.5,0.55,0.5,"data [b]");
  cpgmove(6.0,0.0);  
  cpgdraw(6.0,1.0);
  cpgmtxt("B",-3.0,0.65,0.5,"Show");
  cpgmtxt("B",-1.5,0.65,0.5,"coords [c]");
  cpgmove(7.0,0.0);  
  cpgdraw(7.0,1.0);
  cpgmtxt("B",-3.0,0.75,0.5,"Nullify [n]");
  cpgmtxt("B",-1.5,0.75,0.5,"Denullify [d]");
  cpgmove(8.0,0.0);  
  cpgdraw(8.0,1.0);
  cpgmtxt("B",-1.5,0.85,0.5,"Next plot [q]");
  cpgmove(9.0,0.0);  
  cpgdraw(9.0,1.0);
  cpgmtxt("B",-3.0,0.95,0.5,"Ultimate");
  cpgmtxt("B",-1.5,0.95,0.5,"Quit [Q]");
  
  // Get ready to plot graph
  cpgsvp (0.2, 0.8, 0.3, 0.9);/*set viewport- XLEFT,XRIGHT,YBOT,YTOP*/
  last_params->do_cpgswin();
  cpgbox("BCNST",0.0,0,"BCNST",0.0,0);
  
  string title = base_title + make_string(last_params->get_plot_in_hierachy());

  if( inputs[0]->get_domain().substr(0,7)=="Fourier" && xlabel == "Time (s)" )
    xlabel = "Frequency (Hz)";
  else if( inputs[0]->get_domain()=="Time" && xlabel == "Frequency (Hz)" ) 
    xlabel = "Time (s)";

  cpglab(xlabel.c_str(),ylabel.c_str(),title.c_str());

  return true;
}

template<class DataType>
bool dsp::Plotter<DataType>::highlight(float& x, float& y){
  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"Detected a '%c' at x=%f and y=%f\n",
	    user_char,x,y);

  const bool MAKE_GOOD = true;
  const bool MAKE_BAD = false;

  bool direction = MAKE_GOOD; /* to satisfy compiler */
  if( user_char=='n')
    direction = MAKE_BAD;
  if( user_char=='d')
    direction = MAKE_GOOD;

  // 1. work out which zone is to be nullified
  float xref=x;
  float yref=y;
  cpgband(2, 0, xref, yref, &x, &y, &user_char);

  if(x<xref){ float tmp=x; x=xref; xref=tmp; }
  if(y<yref){ float tmp=y; y=yref; yref=tmp; }
  
  /* so xref=min & x=max */
  DataType* newie = NULL;
  int status_to_left,status_to_right,orig_data_status;

  for(int dataset=0;dataset< int(plotdatas.size()); dataset++){
    if( newie ){
      delete newie; newie = 0;
    }
    
    orig_data_status = plotdatas[dataset]->get_data_is_good();

    /* do the lhs split */
    status_to_left   = plotdatas[dataset]->get_data_is_good();

    if(direction==MAKE_BAD)
      status_to_right = false;
    else
      status_to_right = true;

    fprintf(stderr,
	    "dataset %d: calling split 1st time with lhs limit=%f\n",
	    dataset,xref);

    // if successful, newie will come out before xpos
    if( plotdatas[dataset]->split(newie,xref,status_to_left,status_to_right) ){
      DataType* pooey = new DataType;
      pooey->operator=( *newie );
      plotdatas.insert(plotdatas.begin()+dataset,pooey);
      fprintf(stderr,
	      "highlight:: have inserted newie before dataset %d\n",
	      dataset);
      dataset++;
    }

    /* do the rhs split */
    status_to_left = plotdatas[dataset]->get_data_is_good();
    status_to_right = orig_data_status;
    
    if( newie ){
      delete newie; newie = 0;
    }

    fprintf(stderr,"calling split 2nd time with rhs limit=%f\n",x);
    if( plotdatas[dataset]->split(newie,x,status_to_left,status_to_right) ){
      DataType* pooey = new DataType;
      pooey->operator=( *newie );
      plotdatas.insert(plotdatas.begin()+dataset,pooey);
      fprintf(stderr,"inserted before dataset %d\n",dataset);
      dataset++;
    }
  }

  fprintf(stderr,"Ready to merge\n");

  for( int idataset=0; idataset< int(plotdatas.size()-1); idataset++){
    if( plotdatas[idataset]->merge(*plotdatas[idataset+1]) ){
      plotdatas.erase(plotdatas.begin()+idataset+1);
      fprintf(stderr,"erased dataset %d\n",idataset+1);
      idataset--;
    }
    else
      fprintf(stderr,"datasets %d and %d not merged\n",idataset,idataset+1);
  }

  return true;
}

#endif // !defined(__Plotter_h)
