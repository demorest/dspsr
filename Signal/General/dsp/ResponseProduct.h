//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/ResponseProduct.h,v $
   $Revision: 1.2 $
   $Date: 2004/11/01 16:49:49 $
   $Author: wvanstra $ */

#ifndef __ResponseProduct_h
#define __ResponseProduct_h

#include "dsp/Response.h"
#include "Reference.h"

namespace dsp {

  //! Represents a product of Response instances
  class ResponseProduct: public Response {

  public:

    //! Default constructor
    ResponseProduct ();

    //! Destructor
    ~ResponseProduct ();

    //! Create a product to match the input
    void match (const Observation* input, unsigned nchan);

    //! Create a product to match the response
    void match (const Response* response);

    //! Add a response to the product
    void add_response (Response* response);

  protected:

    //! The responses
    vector< Reference::To<Response> > response;

    //! The builder
    void build ();

  };
  
}

#endif
