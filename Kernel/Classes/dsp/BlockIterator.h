//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/BlockIterator.h

#ifndef __BlockIterator_h
#define __BlockIterator_h

//! An iterator that strides non-contiguous blocks of data
template <typename T>
class BlockIterator
{
 public:

  //! Construct from base pointer
  inline BlockIterator (T* start_of_first_block)
    {
      current = start_of_first_block;
      end_of_data = 0;
      
      data_size = 0;
      block_size = 0;
      increment = 1;
    }
  
  inline BlockIterator (const BlockIterator& copy)
    {
      operator = (copy);

      data_size = copy.data_size;
      block_size = copy.block_size;
      increment = copy.increment;
    }

  inline const BlockIterator& operator = (const BlockIterator& copy)
  {
    // data_size, block_size and increment are set only by copy constructor

    current = copy.current;
    end_of_data = copy.end_of_data;
    return *this;
  }
  
  void set_data_size (unsigned size)
  {
    data_size = size;
    
    if (data_size > 1)
      end_of_data = current + data_size;
  }

  unsigned get_data_size () const { return data_size; }
  
  void set_block_size (unsigned size)
  {
    block_size = size;
  }

  unsigned get_block_size () const { return block_size; }

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

    if (end_of_data && current == end_of_data)
    {
      end_of_data += block_size;
      current += block_size - data_size;
    }
  }

  inline T operator * () const
  {
    return *current;
  }

 protected:

  T* current;
  T* end_of_data;

  unsigned data_size;
  unsigned block_size;
  unsigned increment;

};

#endif
