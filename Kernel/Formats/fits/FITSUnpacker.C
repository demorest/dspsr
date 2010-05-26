/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "assert.h"

#include "dsp/FITSUnpacker.h"
#include "Error.h"

#define ONEBIT_MASK 0x1
#define TWOBIT_MASK 0x3
#define FOURBIT_MASK 0xf

#define ONEBIT_SCALE 0.5
#define FOURBIT_SCALE 7.5
#define EIGHTBIT_SCALE 31.5



using std::cerr;
using std::endl;
using std::vector;
using std::cout;

const int BYTE_SIZE = 8;

dsp::FITSUnpacker::FITSUnpacker(const char* name) : HistUnpacker(name) {}

/**
 * @brief Iterate each row (subint) and sample extracting the values
 *        from input buffer and placing the scaled value in the appropriate
 *        position address by 'into'.
 * @throws InvalidState if nbit != 1, 2, 4 or 8.
 */

void dsp::FITSUnpacker::unpack()
{
  if (verbose) {
    cerr << "dsp::FITSUnpacker::unpack" << endl;
  }

  // Determine which mapping function to use depending on how many
  // samples exist per byte.
  float (*bitNumber)(int) = NULL;

  const unsigned nbit = input->get_nbit();
  switch (nbit) {
    case 1:
      bitNumber = &oneBitNumber;
      break;
    case 2:
      bitNumber = &twoBitNumber;
      break;
    case 4:
      bitNumber = &fourBitNumber;
      break;
    case 8:
      bitNumber = &eightBitNumber;
      break;
    default:
      throw Error(InvalidState, "FITSUnpacker::unpack",
          "invalid nbit=%d", nbit);
  }

  const unsigned npol  = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndat  = input->get_ndat();

  // Number of of samples in a byte.
  const int samples_per_byte = BYTE_SIZE / nbit;
  const int mod_offset = samples_per_byte - 1;

  const unsigned char* from = input->get_rawptr();

  // Iterate through input data, split the byte depending on number of 
  // samples per byte, get corresponding mapped value and store it
  // as pol-chan-dat.
  //
  // TODO: Use a lookup table.
  for (unsigned idat = 0; idat < ndat; ++idat) {
    for (unsigned ipol = 0; ipol < npol; ++ipol) {
      for (unsigned ichan = 0; ichan < nchan;) {

        const int mod = mod_offset - (ichan % samples_per_byte);
        const int shifted_number = *from >> (mod * nbit);

        float* into = output->get_datptr(ichan, ipol) + idat;
        *into = bitNumber(shifted_number);

        // Move to next byte when the entire byte has been split.
        if ((++ichan) % (samples_per_byte) == 0) {
          ++from;
        }
      }
    }
  }
}


bool dsp::FITSUnpacker::matches(const Observation* observation)
{
    return observation->get_machine() == "FITS";
}


/**
 * @brief Mask and scale ( - 0.5) a one-bit number.
 * @param int Contains an unsigned one-bit number to be masked and scaled
 * @return Scaled one-bit value.
 */

float oneBitNumber(const int num)
{
    const int masked_number = num & ONEBIT_MASK;
    return masked_number - ONEBIT_SCALE;
}


/**
 * @brief Scale the eight-bit number ( - 31.5).
 * @param int Eight-bit number to be scaled
 * @return Scaled eight-bit value.
 */

float eightBitNumber(const int num)
{
    return num - EIGHTBIT_SCALE;
}


/**
 * @brief Mask (0xf) and scale ( - 7.5) an unsigned four-bit number.
 * @param int Contains the unsigned four-bit number to be scaled.
 * @returns float Scaled four-bit value.
 */

float fourBitNumber(const int num)
{
    const int masked_number = num & FOURBIT_MASK;
    return masked_number - FOURBIT_SCALE;
}


/**
 * @brief Mask (0x3) and scale an unsigned two-bit number:
 *        0 = -2
 *        1 = -0.5
 *        2 = 0.5
 *        3 = 2.0
 *
 * @param int Contains the unsigned two-bit number to be scaled.
 * @returns float Scaled two-bit value.
 */

float twoBitNumber(const int num)
{
    const int masked_number = num & TWOBIT_MASK;
    switch (masked_number) {
        case 0:
            return -2.0;
        case 1:
            return -0.5;
        case 2:
            return 0.5;
        case 3:
            return 2.0;
        default:
            return 0.0;
    }
}

