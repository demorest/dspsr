//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/StepIterator.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/10 11:11:04 $
   $Author: straten $ */

#ifndef __dsp_StepIterator_h
#define __dsp_StepIterator_h

//! An iterator through contiguous data
template <typename T>
class StepIterator
{
 public:

  //! Construct from base pointer
  StepIterator (T* start)
    {
      current = start;
      increment = 1;
    }
  
  StepIterator (const StepIterator& copy)
    {
      operator = (copy);
      increment = copy.increment;
    }

  const StepIterator& operator = (const StepIterator& copy)
  {
    // increment is set only by copy constructor
    current = copy.current;
  }

  void set_increment (unsigned step)
  {
    increment = step;
  }

  void* ptr ()
  {
    return current;
  }

  void operator ++ ()
  {
    current += increment;
  }

  T operator * ()
  {
    return *current;
  }

 protected:

  T* current;
  unsigned increment;

};

#endif
