#include "TwoBitTable.h"

dsp::TwoBitTable::TwoBitTable (Type type)
{
  float offset_binary[4]   = {-0.75, -0.25, 0.25, 0.75};
  float sign_magnitude[4]  = {0.25, 0.75, -0.25, -0.75};
  float twos_complement[4] = {0.25, 0.75, -0.75, -0.25};

  float* voltages = 0;

  switch (type) {

  case OffsetBinary:
    voltages = offset_binary;
    break;
    
  case SignMagnitude:
    voltages = sign_magnitude;
    break;
    
  case TwosComplement:
    voltages = twos_complement;
    break;

  default:
    throw "TwoBitTable:: unsupported type";
  }

  // 256 possible unsigned chars * 4 floats per bytes
  table = new float [ 256 * 4 ];

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

dsp::TwoBitTable::~TwoBitTable ()
{
  delete [] table;
}


