//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Switcher_h_
#define __Switcher_h_

/*!

  This class takes 2 buffers, and basically swaps them.

*/

#include "Reference.h"

#include "dsp/TimeSeries.h"
#include "dsp/Operation.h"

namespace dsp {

  class Switcher : public Operation {
    public :
      
      //! Constructor
      Switcher();
    
    //! Virtual destructor
    virtual ~Switcher();

    //! Set the buffer that has data beforehand and no data afterwards
    virtual void set_to_empty(TimeSeries* _to_empty){ to_empty = _to_empty; }

    //! Retrieve a pointer to the buffer that has data beforehand and no data afterwards
    virtual TimeSeries* get_to_empty(){ return to_empty; }

    //! Set the buffer that has no data beforehand and data afterwards
    virtual void set_to_fill(TimeSeries* _to_fill){ to_fill = _to_fill; }

    //! Retrieve a pointer to the buffer that has no data beforehand and data afterwards
    virtual TimeSeries* get_to_fill(){ return to_fill; }

  protected:

    //! Does the switching
    virtual void operation();

    //! The buffer that has no data beforehand and data afterwards
    Reference::To<TimeSeries> to_empty;

    //! The buffer that has data beforehand and no data afterwards
    Reference::To<TimeSeries> to_fill;

  };

}

#endif
