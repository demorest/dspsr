#include "TwoBitTable.h"

dsp::TwoBitTable::TwoBitTable (Type type)
{
  // 256 possible unsigned chars * 4 floats per bytes
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

  float* tabval = table;

  unsigned char bits = 0x00;
  unsigned char mask = 0x03;

  for (int val=0; val<256; val++) {
    // the most significant two bits are considered to be first
    for (int shift=6; shift >= 0; shift-=2) {
      *tabval = voltages[(bits>>shift)&mask];
      tabval ++;
    }
    bits ++;
  }
}

void dsp::TwoBitTable::four_vals (float* vals, Type type, float lo, float hi)
{
  switch (type) {

  case OffsetBinary:
    vals[0x00] = -hi;
    vals[0x01] = -lo;
    vals[0x10] = lo;
    vals[0x11] = hi;
    break;
    
  case SignMagnitude:
    vals[0x00] = lo;
    vals[0x01] = hi;
    vals[0x10] = -lo;
    vals[0x11] = -hi;
    break;
    
  case TwosComplement:
    vals[0x00] = lo;
    vals[0x01] = hi;
    vals[0x10] = -hi;
    vals[0x11] = -lo;
    break;

  default:
    throw "TwoBitTable::four_vals unsupported type";
  }
}
