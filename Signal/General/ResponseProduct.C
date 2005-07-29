#include "dsp/ResponseProduct.h"

dsp::ResponseProduct::ResponseProduct ()
{
}

//! Destructor
dsp::ResponseProduct::~ResponseProduct ()
{
}

//! Create a product to match the input
void dsp::ResponseProduct::match (const Observation* obs, unsigned nchan)
{
  for (unsigned iresp=0; iresp < response.size(); iresp++)
    response[iresp]->match (obs, nchan);

  build ();
}

//! Create a product to match the response
void dsp::ResponseProduct::match (const Response* _response)
{
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

void dsp::ResponseProduct::build ()
{
  if (response.size() == 0)
    throw Error (InvalidState, "dsp::ResponseProduct::build",
		 "no responses in product");

  if (!component_changed)
    return;

  throw Error (InvalidState, "dsp::ResponseProduct::build",
	       "in a state of transition");

  for (unsigned iresp=1; iresp < response.size(); iresp++)
    response[iresp]->match (response[0]);

  Response::operator = (*response[0]);
  for (unsigned iresp=1; iresp < response.size(); iresp++)
     Response::operator *= (*response[iresp]);
}
