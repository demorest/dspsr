//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/StepIterator.h

#ifndef __dsp_StepIterator_h
#define __dsp_StepIterator_h

//! An iterator through contiguous data
template <typename T>
class StepIterator
{
 public:

  //! Construct from base pointer
  inline StepIterator (T* start)
    {
      current = start;
      increment = 1;
    }
  
  inline StepIterator (const StepIterator& copy)
    {
      operator = (copy);
      increment = copy.increment;
    }

  inline const StepIterator& operator = (const StepIterator& copy)
  {
    // increment is set only by copy constructor
    current = copy.current;
    return *this;
  }

  void set_increment (unsigned step)
  {
    increment = step;
  }

  const void* ptr ()
  {
    return current;
  }

  inline void operator ++ ()
  {
    current += increment;
  }

  inline T operator * ()
  {
    return *current;
  }

 protected:

  T* current;
  unsigned increment;

};

#endif
