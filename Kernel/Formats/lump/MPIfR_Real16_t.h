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




// MPIfR_Real16_t.h
// class to deal with conversion to a 16 bit real
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

#ifndef MPIfR_Real16_t_H
#define MPIfR_Real16_t_H

// INCLUDES
#ifndef __STDC_CONSTANT_MACROS
#  define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#  define __STDC_LIMIT_MACROS
#endif
#ifndef _ISOC99_SOURCE
#  define _ISOC99_SOURCE
#endif
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE 1
#endif
#ifndef __USE_ISOC99
#  define __USE_ISOC99 1
#endif
#ifndef _ISOC99_SOURCE
#  define _ISOC99_SOURCE
#endif
#ifndef __USE_MISC
#  define __USE_MISC 1
#endif
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include <inttypes.h>
#include <limits>
#ifdef __cplusplus
#  include <cstddef>
#else
#  include <stddef.h>
#endif
#include <stdint.h>
// we want to use ISO C9X stuff
// we want to use some GNU stuff
// But this sometimes breaks time.h
#include <time.h>






typedef float                       Real32_t;
typedef double                      Real64_t;



/* restrict
   This is a really useful modifier, but it is not supported by
   all compilers.  Furthermore, the different ways to specify it
   (double * restrict dp0, double dp1[restrict]) are not available
   in the same release of a compiler.  If you are still using an old
   compiler, your performace is going to suck anyway, so this code
   will only give you restrict when it is fully available.
*/
#ifdef __GNUC__
#  ifdef restrict
/*   Someone else has already defined it.  Hope they got it right. */
#  elif !defined(__GNUG__) && (__STDC_VERSION__ >= 199901L)
/*   Restrict already available */
#  elif !defined(__GNUG__) && (__GNUC__ > 2) || (__GNUC__ == 2 && __GNUC_MINOR__ >= 95)
#    define restrict __restrict
#  elif (__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#    define restrict __restrict
#  else
#    define restrict
#  endif
#else
#  ifndef restrict
#    define restrict
#  endif
#endif





// set up a namespace area for stuff.
namespace MPIfR {

namespace DATA_TYPE {



//_CLASS  Real16_t
class Real16_t {
//_DESC  full description of class

//_FILE  files used and logical units used

//_LIMS  design limitations

//_BUGS  known bugs

//_CALL  list of calls

//_KEYS  

//_HIST  DATE NAME PLACE INFO

//_END


// NAMESPACE ISSUES    


public:
    Real16_t() throw() {};
    Real16_t(const Real32_t& restrict f) throw() {bits = Real16_t(&f).bits;}
    Real16_t(const Real32_t* const restrict fp) throw();
    Real16_t(const Real64_t& restrict f) throw() {bits = Real16_t(&f).bits;}
    Real16_t(const Real64_t* const restrict fp) throw();
    Real16_t(const int_fast64_t& restrict i) throw() {Real64_t f(i); bits = Real16_t(&f).bits;}
    Real16_t(const int_fast64_t* const restrict ip) throw() {Real64_t f(*ip); bits = Real16_t(&f).bits;}
    Real16_t(const uint_fast64_t& restrict i) throw() {Real64_t f(i); bits = Real16_t(&f).bits;}
    Real16_t(const uint_fast64_t* const restrict ip) throw() {Real64_t f(*ip); bits = Real16_t(&f).bits;}

    Real32_t to_Real32_t() const throw();
    Real64_t to_Real64_t() const throw();

    Real16_t& load_bits(uint16_t b) throw() {bits = b; return *this;}
    Real16_t& load_bits_byteswap(uint16_t b) throw() {bits = ((b>>8)|(b<<8)); return *this;}


    
protected:



private:
    uint16_t bits;


    
};


// CLASS FUNCTIONS



// HELPER FUNCTIONS



}  // end namespace DATA_TYPE

}  // end namespace MPIfR

#endif // MPIfR_Real16_t_H
