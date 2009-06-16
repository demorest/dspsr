/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseProduct.h"

using namespace std;

dsp::ResponseProduct::ResponseProduct ()
{
  match_index = copy_index = 0;
}

//! Destructor
dsp::ResponseProduct::~ResponseProduct ()
{
}

//! Create a product to match the input
void dsp::ResponseProduct::match (const Observation* obs, unsigned nchan)
{
  if (verbose)
    cerr << "dsp::ResponseProduct::match (const Observation*)" << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
    response[iresp]->match (obs, nchan);

  build ();
}

//! Create a product to match the response
void dsp::ResponseProduct::match (const Response* _response)
{
  if (verbose)
    cerr << "dsp::ResponseProduct::match (const Response*)" << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
    response[iresp]->match (_response);

  build ();
}

//! Add a response to the product
void dsp::ResponseProduct::add_response (Response* _response)
{
  response.push_back (_response);
  _response->changed.connect (this, &ResponseProduct::set_component_changed);
}

    //! Called when a component has changed
void dsp::ResponseProduct::set_component_changed (const Response& response)
{
  component_changed = true;
}

    //! Set the element that is copied
void dsp::ResponseProduct::set_copy_index (unsigned i)
{
  copy_index = i;
}

//! Set the element to which all others are matched
void dsp::ResponseProduct::set_match_index (unsigned i)
{
  match_index = i;
}

using namespace std;

void dsp::ResponseProduct::build ()
{
  if (verbose)
    cerr << "dsp::ResponseProduct::build" << endl;

  if (response.size() == 0)
    throw Error (InvalidState, "dsp::ResponseProduct::build",
		 "no responses in product");

  //if (!component_changed)
  //  return;

  if (verbose)
    cerr  << "dsp::ResponseProduct::build match_index=" << match_index << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
    if (iresp != match_index)
      response[iresp]->match (response[match_index]);

  if (verbose)
    cerr << "dsp::ResponseProduct::build copy_index=" << copy_index << endl;

  Response::operator = (*response[copy_index]);

  if (verbose)
    cerr << "dsp::ResponseProduct::build ndat=" << ndat 
         << " nchan=" << nchan << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
    if (iresp != copy_index)
      Response::operator *= (*response[iresp]);

  if (verbose)
    cerr << "dsp::ResponseProduct::build DONE!" << endl;
}

