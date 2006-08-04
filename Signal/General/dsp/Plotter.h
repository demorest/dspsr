//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Plotter_h
#define __dsp_Plotter_h

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <memory>

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

    void set_just_do_basics(bool _just_do_basics)
    { just_do_basics = _just_do_basics; }    
    bool get_just_do_basics(){ return just_do_basics; } 

    //! Decide whether to show the plot number at top of screen
    void set_show_plot_number(bool _show_plot_number)
    { show_plot_number = _show_plot_number; }    
    //! Inquire whether to show the plot number at top of screen
    bool get_show_plot_number(){ return show_plot_number; } 

    bool disable_viewer_interaction;

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

    //! When set to true, a call to plot() just does BasicPlotter::plot()
    bool just_do_basics;

    //! If true, (plot XXX) is shown as part of title [true]
    bool show_plot_number;

  };

}

template<class DataType>
dsp::Plotter<DataType>::Plotter() : BasicPlotter<DataType>() {
  Operation::set_name("Plotter");

  cpgopen_response = "101/xs";
  dev = -1;
  
  base_title = " ";
  xlabel = "Time (s)";
  ylabel = "Intensity";

  just_do_basics = false;
  show_plot_number = true;
  disable_viewer_interaction = false;

  BasicPlotter<DataType>::mutated = true;
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

  char value[4096];
  int length = 4095;
  cpgqinf("DEV/TYPE",value, &length);
  string plot_device = string(value,value+length);
  string fourchars = plot_device.substr(plot_device.size()-4,4);

  if( fourchars == "/png" || fourchars=="/PNG" )
    disable_viewer_interaction = true;
  
  return true;
}

template<class DataType>
bool dsp::Plotter<DataType>::plot(){
  if( just_do_basics )
    return dsp::BasicPlotter<DataType>::plot();

  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"In dsp::Plotter::plot()\n\n\n\n\n\n\n\n");

  if( !this->EnsureReadiness() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to ensure readiness\n");
    return false;
  }

  if( !open_plotwindow() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to open plot window\n");
    return false;
  }

  if( !this->set_plotdatas() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to set plotdatas\n");
    return false;
  }
  if( BasicPlotter<DataType>::plotdatas.size()==0 ){
    fprintf(stderr,"Error!  No plotdatas specified\n");
    return false;
  }
  if( !this->set_params() ){
    fprintf(stderr,"dsp::Plotter::plot() failed to set params\n");
    return false;
  }  

  BasicPlotter<DataType>::user_char = '\0';
  while( !(BasicPlotter<DataType>::user_char=='q' || BasicPlotter<DataType>::user_char=='Q') ){
    fprintf(stderr,"\nContinued in loop as user_char='%c'\n",BasicPlotter<DataType>::user_char);
    
    if( BasicPlotter<DataType>::mutated ){
      plot_background();
      for( unsigned iplotdata=0; iplotdata<BasicPlotter<DataType>::plotdatas.size(); iplotdata++){
	cpgsci(1);
	if( BasicPlotter<DataType>::plotdatas.size() > 1 )
	  this->set_colour_index(iplotdata);
	if( Ready::verbose || Operation::verbose )
	  fprintf(stderr,"dsp::Plotter::plot() calling plotdatas[%d]->plot()\n",
		  iplotdata);
	BasicPlotter<DataType>::plotdatas[iplotdata]->plot(BasicPlotter<DataType>::params.back());
      }
    }

    cpgsci(1);
    cpgsls(1);
    cpgsch(1.0);
    if( disable_viewer_interaction )
      break;
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
  
  plot::PlotParams p = params.back();

  // Used to store x co-ord of cursor upon keypress
  float x = 0.5*(p.get_xmax() - p.get_xmin());
  // Used to store y co-ord of cursor upon keypress      
  float y = 0.5*(p.get_ymax() - p.get_ymin());
  float xref;       // Used to store secondary x co-ord of cursor upon e.g. completion of zoom box
  float yref;       // Used to store secondary y co-ord of cursor upon e.g. completion of zoom box
 
#if defined ( __GNUC__ ) && ( __GNUC__ >= 3 ) && ( __GNUC_MINOR__ >= 4 )
  char& user_char_copy = BasicPlotter<DataType>::user_char;
#else
  char& user_char_copy = user_char;
#endif

  if(cpgcurs(&x,&y,&user_char_copy)==0){
    cerr<<"Error in cpgcurs\n";
    return false;
  }

  if(Ready::verbose || Operation::verbose)
    cerr<<"After cpgcurs user_char is '"<<BasicPlotter<DataType>::user_char<<"'\n\n";
  
  ///////////////////////////////////////////////////////////
  // Mouseclick to draw a zoom-box
  if(BasicPlotter<DataType>::user_char=='z' || BasicPlotter<DataType>::user_char=='A'){
    if(Ready::verbose || Operation::verbose)
	cerr<<"Detected a 'z' at x='"<<x<<"' and y='"<<y<<"'\n";
    xref=x;
    yref=y;

    cpgband(2, 0, xref, yref, &x, &y, &user_char_copy);

    if(x<xref){ float tmp=x; x=xref; xref=tmp; }
    if(y<yref){ float tmp=y; y=yref; yref=tmp; }

    BasicPlotter<DataType>::params.push_back( plot::PlotParams(xref, x, yref, y, BasicPlotter<DataType>::params.back()) );

    BasicPlotter<DataType>::mutated = true;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // highlight data
  /*
  if(BasicPlotter<DataType>::user_char=='n' || BasicPlotter<DataType>::user_char=='d' ){
    highlight(x,y);
    BasicPlotter<DataType>::params.push_back( BasicPlotter<DataType>::params.back() );
    BasicPlotter<DataType>::params.back().incr();
    
    BasicPlotter<DataType>::mutated = true;
    return true;
  }
  */  

  ///////////////////////////////////////////////////////////
  // Undo to previous level
  if(BasicPlotter<DataType>::user_char=='u' && BasicPlotter<DataType>::params.size()>1){
    BasicPlotter<DataType>::params.pop_back();
    BasicPlotter<DataType>::mutated = true;
    return true;
  }
    
  ///////////////////////////////////////////////////////////
  // Zoom right
  if(BasicPlotter<DataType>::user_char=='r'){
    BasicPlotter<DataType>::params.push_back(plot::PlotParams(x,
				x+BasicPlotter<DataType>::params.back().get_xmax()-BasicPlotter<DataType>::params.back().get_xmin(),
				BasicPlotter<DataType>::params.back().get_ymin(),
				BasicPlotter<DataType>::params.back().get_ymax(),
				BasicPlotter<DataType>::params.back()));
    BasicPlotter<DataType>::mutated = true;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // Zoom left
  if(BasicPlotter<DataType>::user_char=='l'){
    BasicPlotter<DataType>::params.push_back(plot::PlotParams(x-(BasicPlotter<DataType>::params.back().get_xmax()-BasicPlotter<DataType>::params.back().get_xmin()),
				x,
				BasicPlotter<DataType>::params.back().get_ymin(),
				BasicPlotter<DataType>::params.back().get_ymax(),
				BasicPlotter<DataType>::params.back()));
    BasicPlotter<DataType>::mutated = true;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // Undo to top level
  if(BasicPlotter<DataType>::user_char=='o'){
    BasicPlotter<DataType>::params.erase(BasicPlotter<DataType>::params.begin()+1,BasicPlotter<DataType>::params.end());
    BasicPlotter<DataType>::mutated = true;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // Block new data
  /*
  if(BasicPlotter<DataType>::user_char=='b'){
    BasicPlotter<DataType>::params.back().switch_block_new_data();
    BasicPlotter<DataType>::mutated = false;
    return true;
  }
  */

  ///////////////////////////////////////////////////////////
  // Print out coordinates
  if(BasicPlotter<DataType>::user_char=='c'){
    fprintf(stderr,"x=%f\ty=%f\n",x,y);
    BasicPlotter<DataType>::mutated = false;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // Output to hardcopy
  if(BasicPlotter<DataType>::user_char=='p'){
    string response = string("dsp_Plotter") + string(".cps/cps");;
    
    fprintf(stderr,"Will ps to '%s'\n",response.c_str());
    
    int psdev = cpgopen(response.c_str());

    if( psdev <= 0 ){
      fprintf(stderr,"dsp::Plotter::open_plotwindow() cpgopen call failed\n");
      return false;
    }
  
    plot_background();
    for( unsigned iplotdata=0; iplotdata<BasicPlotter<DataType>::plotdatas.size(); iplotdata++){
      cpgsci(1);
      //set_colour_index(iplotdata);
      BasicPlotter<DataType>::plotdatas[iplotdata]->plot(BasicPlotter<DataType>::params.back());
    }
   
    cpgclos();
    cpgslct(dev);

    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"Out of replot for ps file\n");

    BasicPlotter<DataType>::mutated = false;
    return true;
  }

  ///////////////////////////////////////////////////////////
  // Quit out
  if( BasicPlotter<DataType>::user_char=='q' || BasicPlotter<DataType>::user_char=='Q' ){
    return true;
  }

  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"You typed '%c'\n",BasicPlotter<DataType>::user_char);
  BasicPlotter<DataType>::mutated = false;

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
  
  plot::PlotParams* last_params = &BasicPlotter<DataType>::params.back();

  /*
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
  */

  // Get ready to plot graph
  //cpgsvp (0.2, 0.8, 0.3, 0.9);/*set viewport- XLEFT,XRIGHT,YBOT,YTOP*/
  cpgsvp( 0.1,0.9,0.1,0.9);

  last_params->do_cpgswin();
  cpgbox("BCNST",0.0,0,"BCNST",0.0,0);
  
  string title = base_title;
  if( show_plot_number )
    title += "  (plot " + make_string(last_params->get_plot_in_hierachy()) + ")";

  if( BasicPlotter<DataType>::inputs[0]->get_domain().substr(0,7)=="Fourier" && xlabel == "Time (s)" )
    xlabel = "Frequency (Hz)";
  else if( BasicPlotter<DataType>::inputs[0]->get_domain()=="Time" && xlabel == "Frequency (Hz)" ) 
    xlabel = "Time (s)";

  cpglab(xlabel.c_str(),ylabel.c_str(),title.c_str());

  return true;
}

template<class DataType>
bool dsp::Plotter<DataType>::highlight(float& x, float& y){
  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"Detected a '%c' at x=%f and y=%f\n",
	    BasicPlotter<DataType>::user_char,x,y);

  const bool MAKE_GOOD = true;
  const bool MAKE_BAD = false;

  bool direction = MAKE_GOOD; /* to satisfy compiler */
  if( BasicPlotter<DataType>::user_char=='n')
    direction = MAKE_BAD;
  if( BasicPlotter<DataType>::user_char=='d')
    direction = MAKE_GOOD;

  // 1. work out which zone is to be nullified
  float xref=x;
  float yref=y;

#if defined ( __GNUC__ ) && ( __GNUC__ >= 3 ) && ( __GNUC_MINOR__ >= 4 )
  char& user_char2 = BasicPlotter<DataType>::user_char;
#else
  char& user_char2 = user_char;
#endif

  cpgband(2, 0, xref, yref, &x, &y, &user_char2);

  if(x<xref){ float tmp=x; x=xref; xref=tmp; }
  if(y<yref){ float tmp=y; y=yref; yref=tmp; }
  
  /* so xref=min & x=max */
  int status_to_left,status_to_right,orig_data_status;

  for(int dataset=0;dataset< int(BasicPlotter<DataType>::plotdatas.size()); dataset++){
    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"Working with dataset %d/%d\n",
	      dataset+1,BasicPlotter<DataType>::plotdatas.size());

    orig_data_status = BasicPlotter<DataType>::plotdatas[dataset]->get_data_is_good();

    /* do the lhs split */
    status_to_left   = BasicPlotter<DataType>::plotdatas[dataset]->get_data_is_good();

    if(direction==MAKE_BAD)
      status_to_right = false;
    else
      status_to_right = true;

    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"dataset %d: calling split 1st time with lhs limit=%f\n",
	      dataset,xref);

    // if successful, newie will come out before xpos
    void* ans = BasicPlotter<DataType>::plotdatas[dataset]->split(xref,status_to_left,status_to_right);
    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"Got ans=%p\n",ans);
    exit(0);

    Reference::To<DataType> newie( dynamic_cast<DataType*>(BasicPlotter<DataType>::plotdatas[dataset]->split(xref,status_to_left,status_to_right).ptr()) );
    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"dsp::Plotter::highlight():: out of split with newie.ptr()=%p\n",newie.ptr());

    if( newie.ptr() ){
      if( Operation::verbose )
	fprintf(stderr,"Successfully out of plotdatas[%d]->split() with newie=%p\n",dataset,newie.get());
      BasicPlotter<DataType>::plotdatas.insert(BasicPlotter<DataType>::plotdatas.begin()+dataset,newie->clone());
      if(Ready::verbose || Operation::verbose)
	fprintf(stderr,"highlight:: have inserted newie before dataset %d\n",
	      dataset);
      dataset++;
    }

    /* do the rhs split */
    status_to_left = BasicPlotter<DataType>::plotdatas[dataset]->get_data_is_good();
    status_to_right = orig_data_status;
    
    if( newie )
      newie = new DataType;

    if(Ready::verbose || Operation::verbose)
      fprintf(stderr,"calling split 2nd time with rhs limit=%f and plotdatas.size()=%d and dataset=%d\n",
	      x,BasicPlotter<DataType>::plotdatas.size(),dataset);
    newie = dynamic_cast<DataType*>( BasicPlotter<DataType>::plotdatas[dataset]->split(x,status_to_left,status_to_right).ptr() );

    if( newie.ptr() ){
      BasicPlotter<DataType>::plotdatas.insert(BasicPlotter<DataType>::plotdatas.begin()+dataset,newie->clone());
      if(Ready::verbose || Operation::verbose)
	fprintf(stderr,"inserted before dataset %d\n",dataset);
      dataset++;
    }
  }

  if(Ready::verbose || Operation::verbose)
    fprintf(stderr,"Ready to merge\n");

  for( int idataset=0; idataset< int(BasicPlotter<DataType>::plotdatas.size()-1); idataset++){
    if( BasicPlotter<DataType>::plotdatas[idataset]->merge(*BasicPlotter<DataType>::plotdatas[idataset+1]) ){
      BasicPlotter<DataType>::plotdatas.erase(BasicPlotter<DataType>::plotdatas.begin()+idataset+1);
      if(Ready::verbose || Operation::verbose)
	fprintf(stderr,"erased dataset %d\n",idataset+1);
      idataset--;
    }
    else
      if(Ready::verbose || Operation::verbose)
	fprintf(stderr,"datasets %d and %d not merged\n",idataset,idataset+1);
  }

  return true;
}

#endif // !defined(__Plotter_h)
