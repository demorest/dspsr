//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/BlockIterator.h,v $
   $Revision: 1.1 $
   $Date: 2008/07/10 07:14:05 $
   $Author: straten $ */

#ifndef __BlockIterator_h
#define __BlockIterator_h

//! An iterator that strides non-contiguous blocks of data
template <typename T>
class BlockIterator
{
 public:

  //! Construct from base pointer
  BlockIterator (T* start_of_first_block)
    {
      current = start_of_data = start_of_first_block;
      end_of_data = 0;
      
      data_size = 0;
      block_size = 0;
      increment = 1;
    }
  
  BlockIterator (const BlockIterator& copy)
    {
      operator = (copy);

      data_size = copy.data_size;
      block_size = copy.block_size;
      increment = copy.increment;
    }

  const BlockIterator& operator = (const BlockIterator& copy)
  {
    // data_size, block_size and increment are set only by copy constructor

    current = copy.current;
    start_of_data = copy.start_of_data;
    end_of_data = copy.end_of_data;
  }
  
  void set_data_size (unsigned size)
  {
    data_size = size;
    
    if (data_size > 1)
      end_of_data = current + data_size;
  }
  
  void set_block_size (unsigned size)
  {
    block_size = size;
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

    if (end_of_data && current == end_of_data)
    {
      start_of_data += block_size;
      end_of_data += block_size;
      current = start_of_data;
    }
  }

  T operator * ()
  {
    return *current;
  }

 protected:

  T* current;
  T* start_of_data;
  T* end_of_data;

  unsigned data_size;
  unsigned block_size;
  unsigned increment;

};

#endif
