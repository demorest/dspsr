#include <stdio.h>
#include <iostream>
#include "TwoBitTable.h"

dsp::TwoBitTable::TwoBitTable (Type type)
{
  // 256 possible unsigned chars * 4 floats per byte
  table = new float [ 256 * 4 ];
  lo_val = 0.25;
  generate (table, type, lo_val, 3.0*lo_val);
}

dsp::TwoBitTable::~TwoBitTable ()
{
  delete [] table;
}

/*!
  \param table pointer to space for at least 4*256=1024 floating point values,
  representing the four outputs for each of 256 possible bytes
  \param type the digitization convention
  \param lo value of voltage in low state
  \param hi value of voltage in high state
*/
void dsp::TwoBitTable::generate (float* table, Type type, float lo, float hi)
{
  float voltages[4];
  dsp::TwoBitTable::four_vals (voltages, type, lo, hi);

#ifdef _DEBUG
  fprintf (stderr, "TwoBitTable::generate: %f %f %f %f\n", 
	   voltages[0],voltages[1],voltages[2],voltages[3]);
#endif

  float* tabval = table;

  unsigned char bits = 0x00;
  unsigned char mask = 0x03;

  for (int val=0; val<256; val++) {
    // the most significant two bits are considered to be first
    for (int shift=6; shift >= 0; shift-=2) {
      *tabval = voltages[(bits>>shift)&mask];
#ifdef _DEBUG
      fprintf (stderr, "%f ", *tabval);
#endif
      tabval ++;
    }
    bits ++;
#ifdef _DEBUG
    fprintf (stderr, "\n");
#endif
  }
}

void dsp::TwoBitTable::four_vals (float* vals, Type type, float lo, float hi)
{
  switch (type) {

  case OffsetBinary:
    //cerr << "TwoBitTable::four_vals Offset Binary" << endl;
    vals[0] = -hi;
    vals[1] = -lo;
    vals[2] = lo;
    vals[3] = hi;
    break;
    
  case SignMagnitude:
    vals[0] = lo;
    vals[1] = hi;
    vals[2] = -lo;
    vals[3] = -hi;
    break;
    
  case TwosComplement:
    vals[0] = lo;
    vals[1] = hi;
    vals[2] = -hi;
    vals[3] = -lo;
    break;

  default:
    throw "TwoBitTable::four_vals unsupported type";
  }
}
