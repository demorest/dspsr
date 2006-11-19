/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SetBufferingPolicy.h"
#include "dsp/TimeSeries.h"

using namespace std;

class Test : public dsp::Transformation<dsp::TimeSeries,dsp::TimeSeries> {

public:

  Test () : dsp::Transformation<dsp::TimeSeries,dsp::TimeSeries>
  ("Test", dsp::anyplace) {}

  void transformation () { cerr << "Test::transformation" << endl; }

};

int main () try 
{

  Test test;

  if ( !test.has_buffering_policy() ) {
    cerr << "test_SetBufferingPolicy policy not set" << endl;
    return -1;
  }

  dsp::BufferingPolicy* policy = test.get_buffering_policy ();

  return 0;

}

catch (Error& error) {
  cerr << "test_SetBufferingPolicy error " << error << endl;
  return -1;
}
