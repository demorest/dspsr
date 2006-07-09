//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/ResponseProduct.h,v $
   $Revision: 1.5 $
   $Date: 2006/07/09 13:27:13 $
   $Author: wvanstra $ */

#ifndef __ResponseProduct_h
#define __ResponseProduct_h

#include "dsp/Response.h"
#include "Reference.h"

namespace dsp {

  //! Represents a product of Response instances
  /*! The dimensions of the product will contain the dimensions of
    each term in the product, as defined by:

   - the largest number of frequency channels
   - the largest dimension: matrix > dual polarization > single; complex > real

  */
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
    std::vector< Reference::To<Response> > response;

    //! Flag set true when a component has changed
    bool component_changed;

    //! Called when a component has changed
    void set_component_changed (const Response& response);

    //! The builder
    void build ();

  };
  
}

#endif
