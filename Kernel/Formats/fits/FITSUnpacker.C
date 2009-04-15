/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

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

dsp::FITSUnpacker::FITSUnpacker(const char* name) : HistUnpacker(name) {}

/**
 * @brief Iterate each row (subint) and sample extracting the values
 *        from input buffer and placing the scaled value in the appropriate
 *        position address by 'into'.
 * @throws InvalidState if nbit != 1, 2, 4 or 8.
 */

void dsp::FITSUnpacker::unpack()
{
    if (verbose)
        cerr << "dsp::FITSUnpacker::unpack" << endl;

    const uint npol = input->get_npol();
    const uint nchan = input->get_nchan();
    const uint nbit = input->get_nbit();
    const uint sampsPerByte = 8 / nbit;

    for (uint idat = 0; idat < input->get_ndat(); ++idat) {
        for (uint ipol = 0; ipol < npol; ++ipol) {
            const unsigned char* from = input->get_rawptr() +
                (idat * nchan * npol / sampsPerByte) +
                (ipol * nchan / sampsPerByte);
            for (uint ichan = 0; ichan < nchan; ++ichan) {
                float* into = output->get_datptr(ichan, ipol);

                // shift the number right (towards LSB) so a mask can 
                // be applied later
                const int mod = (sampsPerByte - 1) - (ichan % sampsPerByte);
                const int shiftedNumber = *from >> (mod * nbit);

                // calculate value depending on nbit
                switch (nbit) {
                    case 1:
                        into[idat] = oneBitNumber(shiftedNumber);
                        break;
                    case 2:
                        into[idat] = twoBitNumber(shiftedNumber);
                        break;
                    case 4:
                        into[idat] = fourBitNumber(shiftedNumber);
                        break;
                    case 8:
                        into[idat] = eightBitNumber(shiftedNumber);
                        break;
                    default:
                        throw Error(InvalidState, "FITSUnpacker::unpack",
                                "invalid nbit=%d", nbit);
                }

                if (ichan % sampsPerByte == 0)
                    ++from;
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

