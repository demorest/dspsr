/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BlockIterator.h"

#include <iostream>
#include <vector>

using namespace std;

int main ()
{
  vector<double> data (300, 0.0);

  unsigned block_size = 23;
  unsigned data_size = 5;

  unsigned current = 0;
  double value = 1.0;

  unsigned count = 0;
  unsigned total = 0;

  while (current < data.size())
  {
    data[current] = value;
    current ++;
    value ++;
    count ++;
    total ++;
    if (count == data_size)
    {
      current += block_size - data_size;
      count = 0;
    }
  }

  BlockIterator<double> iterator ( &(data[0]) );
  iterator.set_data_size (data_size);
  iterator.set_block_size (block_size);

  value = 1.0;

  for (count=0; count < total; count++)
  {
    if (value != *iterator)
      {
	cerr << "BlockIterator fail" << endl;
	cerr << "expected=" << value << " got=" << *iterator << endl;
	return -1;
      }

    value ++;
    ++ iterator;
  }

  cerr << "BlockIterator passes simple test" << endl;

  return 0;
}
