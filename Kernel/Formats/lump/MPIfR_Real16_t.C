//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


// This file is based on a class developed by James M Anderson at the MPIfR.
// The original code has been modified to move in information from other include
// files which have not been copied over.



// MPIfR_Real16_t.cxx
// code to deal with conversion to a 16 bit real
//_HIST  DATE NAME PLACE INFO
// 2011 May 11  James M Anderson  --MPIfR  start



// Copyright (c) 2011, James M. Anderson <anderson@mpifr-bonn.mpg.de>

// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.

// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.





// This class deals with a half-precision binary floating point format,
// from the IEEE 754-2008 16-bit base 2 format, officially referred to as
// binary16.  The class deals with conversion to and from single and double
// precision IEEE 754 base 2 floating point formats.

// See http://en.wikipedia.org/wiki/Half_precision_floating-point_format
//     http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0205j/CIHGAECI.html



// INCLUDES
#include "MPIfR_Real16_t.h"


namespace {
union R32 {
    uint32_t u32;
    Real32_t r32;
};
union R64 {
    uint64_t u64;
    Real64_t r64;
};
}



// set up a namespace area for stuff.
namespace MPIfR {

namespace DATA_TYPE {





// GLOBALS


// FUNCTIONS

Real16_t::
Real16_t(const Real32_t* const restrict fp) restrict throw()
{
    const uint_fast32_t u32 = *(reinterpret_cast<const uint32_t* const restrict>(fp));
    const uint_fast32_t u32_sign = u32 & UINT32_C(0x80000000);
    bits = uint16_t(u32_sign >> 16); // store the sign bit
    // deal with zero
    if((u32 & UINT32_C(0x7FFFFFFF)) == 0) {
        // do nothing
    }
    else {
        const uint_fast32_t u32_expo = u32 & UINT32_C(0x7F800000);
        const uint_fast32_t u32_mant = u32 & UINT32_C(0x007FFFFF);
        // deal with denormal in Real32_t
        if(u32_expo == 0) {
            // make this go to zero, so do nothing
        }
        // deal with +-Inf or NaN, which have all exponent bits set
        else if(u32_expo == UINT32_C(0x7F800000)) {
            bits |= UINT16_C(0x7C00);
            if(u32_mant == 0) {
                // +-Inf
                // do nothing more
            }
            else {
                // NaN, try to take as many of the bits as possible,
                // but make sure that the last bit is on, to keep it a NaN
                bits |= (u32_mant >> 13) | 0x1;
            }
        }
        else {
            // normal number
            // convert the exponent, bias for the binary16 format, then
            // unbiasing for the binary32 format            
            int_fast16_t R16_expo = int_fast16_t(u32_expo >> 23)
                + int_fast16_t(15-127);
            if(R16_expo > 0x1F) {
                // overflow, return +-Inf
                bits |= UINT16_C(0x7C00);
            }
            else if(R16_expo <= 0) {
                // undeflow
                if(R16_expo < -10) {
                    // too far gone, set to zero
                    // do nothing
                }
                else {
                    // add on the leading 1 bit
                    uint_fast16_t R16_mant((u32_mant | UINT32_C(0x00800000)) >> (13-R16_expo));
                    uint_fast16_t round(R16_mant & 0x1);
                    R16_mant >>= 1;
                    if((round)) {
                        ++R16_mant; // rounding up into exponent area is ok
                    }
                    bits |= R16_mant;
                }
            }
            else {
                R16_expo <<= 10;
                uint_fast16_t R16_mant(u32_mant >> 13);
                bits |= R16_expo | R16_mant;
                if((u32_mant & UINT32_C(0x00001000))) {
                    // need to round up, if it goes into exponent, this is ok
                    bits++;
                }
            }
        }
    }
    return;
}



Real16_t::
Real16_t(const Real64_t* const restrict fp) restrict throw()
{
    const uint_fast32_t u32((*(reinterpret_cast<const uint64_t* const restrict>(fp))) >> 32);
    const uint_fast32_t u32_sign = u32 & UINT32_C(0x80000000);
    bits = uint16_t(u32_sign >> 16); // store the sign bit
    // deal with zero
    if((u32 & UINT32_C(0x7FFFFFFF)) == 0) {
        // do nothing
    }
    else {
        const uint_fast32_t u32_expo = u32 & UINT32_C(0x7FF00000);
        const uint_fast32_t u32_mant = u32 & UINT32_C(0x000FFFFF);
        // deal with denormal in Real32_t
        if(u32_expo == 0) {
            // make this go to zero, so do nothing
        }
        // deal with +-Inf or NaN, which have all exponent bits set
        else if(u32_expo == UINT32_C(0x7FF00000)) {
            bits |= UINT16_C(0x7C00);
            if(u32_mant == 0) {
                // +-Inf
                // do nothing more
            }
            else {
                // NaN, try to take as many of the bits as possible,
                // but make sure that the last bit is on, to keep it a NaN
                bits |= (u32_mant >> 10) | 0x1;
            }
        }
        else {
            // normal number
            // convert the exponent, bias for the binary16 format, then
            // unbiasing for the binary64 format            
            int_fast16_t R16_expo = int_fast16_t(u32_expo >> 20)
                + int_fast16_t(15-1023);
            if(R16_expo > 0x1F) {
                // overflow, return +-Inf
                bits |= UINT16_C(0x7C00);
            }
            else if(R16_expo <= 0) {
                // undeflow
                if(R16_expo < -11) {
                    // too far gone, set to zero
                    // do nothing
                }
                else {
                    // add on the leading 1 bit
                    uint_fast16_t R16_mant((u32_mant | UINT32_C(0x00100000)) >> (10-R16_expo));
                    uint_fast16_t round(R16_mant & 0x1);
                    R16_mant >>= 1;
                    if((round)) {
                        ++R16_mant; // rounding up into exponent area is ok
                    }
                    bits |= R16_mant;
                }
            }
            else {
                R16_expo <<= 10;
                uint_fast16_t R16_mant(u32_mant >> 10);
                bits |= R16_expo | R16_mant;
                if((u32_mant & UINT32_C(0x00000200))) {
                    // need to round up, if it goes into exponent, this is ok
                    bits++;
                }
            }
        }
    }
    return;
}


Real32_t Real16_t::
to_Real32_t() const restrict throw()
{
    uint_fast16_t u16_sign(bits & UINT16_C(0x8000));
    uint32_t u32 = uint_fast32_t(u16_sign) << 16;
    // deal with zero
    if((bits & UINT16_C(0x7FFF)) == 0) {
        // do nothing
    }
    else {
        const uint_fast16_t u16_expo = bits & UINT16_C(0x7C00);
              uint_fast16_t u16_mant = bits & UINT16_C(0x03FF);
        // deal with denormal
        if(u16_expo == 0) {
            // what is the exponent?
            // shift over until the leading bit is in the exponent area
            int_fast16_t expo = -1;
            do {
                expo++;
                u16_mant <<= 1;
            }
            while ((u16_mant & UINT16_C(0x0400)) == 0);
            int_fast16_t R32_expo = int_fast16_t(u16_expo >> 10)
                + int_fast16_t(127-15) -expo;
            u32 |= (uint_fast32_t(R32_expo) << 23)
                | ((uint_fast32_t(u16_mant) & UINT16_C(0x03FF)) << 13);
        }
        // deal with +-Inf or NaN, which have all exponent bits set
        else if(u16_expo == UINT16_C(0x7C00)) {
            u32 |= UINT32_C(0x7F800000);
            if(u16_mant == 0) {
                // +-Inf
                // do nothing more
            }
            else {
                // NaN, try to take as many of the bits as possible,
                // but make sure that the last bit is on, to keep it a NaN
                u32 |= uint_fast32_t(u16_mant) << 13;
            }
        }
        else {
            // normal number
            // convert the exponent, bias for the binary32 format, then
            // unbiasing for the binary16 format            
            int_fast16_t R32_expo = int_fast16_t(u16_expo >> 10)
                + int_fast16_t(127-15);
            u32 |= (uint_fast32_t(R32_expo) << 23)
                | (uint_fast32_t(u16_mant) << 13);
        }
    }
    union R32 u;
    u.u32=u32;
    return u.r32;
}

Real64_t Real16_t::
to_Real64_t() const restrict throw()
{
    uint_fast16_t u16_sign(bits & UINT16_C(0x8000));
    uint32_t u32 = uint_fast32_t(u16_sign) << 16;
    // deal with zero
    if((bits & UINT16_C(0x7FFF)) == 0) {
        // do nothing
    }
    else {
        const uint_fast16_t u16_expo = bits & UINT16_C(0x7C00);
              uint_fast16_t u16_mant = bits & UINT16_C(0x03FF);
        // deal with denormal
        if(u16_expo == 0) {
            // what is the exponent?
            // shift over until the leading bit is in the exponent area
            int_fast16_t expo = -1;
            do {
                expo++;
                u16_mant <<= 1;
            }
            while ((u16_mant & UINT16_C(0x0400)) == 0);
            int_fast16_t R32_expo = int_fast16_t(u16_expo >> 10)
                + int_fast16_t(1023-15) -expo;
            u32 |= (uint_fast32_t(R32_expo) << 20)
                | ((uint_fast32_t(u16_mant) & UINT16_C(0x03FF)) << 10);
        }
        // deal with +-Inf or NaN, which have all exponent bits set
        else if(u16_expo == UINT16_C(0x7C00)) {
            u32 |= UINT32_C(0x7FF00000);
            if(u16_mant == 0) {
                // +-Inf
                // do nothing more
            }
            else {
                // NaN, try to take as many of the bits as possible,
                // but make sure that the last bit is on, to keep it a NaN
                u32 |= uint_fast32_t(u16_mant) << 10;
            }
        }
        else {
            // normal number
            // convert the exponent, bias for the binary32 format, then
            // unbiasing for the binary16 format            
            int_fast16_t R32_expo = int_fast16_t(u16_expo >> 10)
                + int_fast16_t(1023-15);
            u32 |= (uint_fast32_t(R32_expo) << 20)
                | (uint_fast32_t(u16_mant) << 10);
        }
    }
    union R64 u;
    u.u64=uint64_t(u32)<<32;
    return u.r64;
}





}  // end namespace DATA_TYPE

}  // end namespace MPIfR


