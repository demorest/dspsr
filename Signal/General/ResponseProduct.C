#include "dsp/ResponseProduct.h"


dsp::ResponseProduct::ResponseProduct ()
{
}

//! Destructor
dsp::ResponseProduct::~ResponseProduct ()
{
}

//! Create a product to match the input
void dsp::ResponseProduct::match (const Observation* input, unsigned nchan)
{
}

//! Create a product to match the response
void dsp::ResponseProduct::match (const Response* response)
{
}

//! Add a response to the product
void dsp::ResponseProduct::add_response (Response* _response)
{
  response.push_back (_response);
}

