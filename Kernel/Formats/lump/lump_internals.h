/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include "machine_endian.h"


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







// while not strictly guaranteed to work by the standard, all sensible compilers
// generate instructions to do the right thing with the unions below.
inline float int32_t_bits_to_float(int32_t a){
  union{
    float f;
    int32_t i;
  }dat;
  dat.i = a;
  return dat.f;
}
inline float uint32_t_bits_to_float(uint32_t a){
  union{
    float f;
    uint32_t i;
  }dat;
  dat.i = a;
  return dat.f;
}
inline double int64_t_bits_to_double(int64_t a){
  union{
    double d;
    int64_t i;
  }dat;
  dat.i = a;
  return dat.d;
}
inline double uint64_t_bits_to_double(uint64_t a){
  union{
    double d;
    uint64_t i;
  }dat;
  dat.i = a;
  return dat.d;
}





// byteswapping floats is tricky, as loading a byteswapped float into an FPU
// register can cause bits to get twiddled (e.g. automatic conversion of
// signalling NaNs to non-signalling NaNs), so until the bits are in their
// proper order, only load into integer registers.
inline int16_t byteswap_int16_t(int16_t i)
{
    // no __builtin_bswap16 in most gcc versions
    uint16_t u(i);
    return int16_t((u>>8)|(u<<8));
}
inline int32_t byteswap_int32_t(int32_t i)
{
#ifdef __GNUC__
#    if ((__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3))
    return __builtin_bswap32(i);
#  else
    uint32_t u(i);
    return int32_t((u>>24)|((u&0xFF0000)>>8)|((u&0xFF00)<<8)|(u<<24));
#  endif
#else
    uint32_t u(i);
    return int32_t((u>>24)|((u&0xFF0000)>>8)|((u&0xFF00)<<8)|(u<<24));
#endif
}
inline int64_t byteswap_int64_t(int64_t i)
{
#ifdef __GNUC__
#    if ((__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3))
    return __builtin_bswap64(i);
#  else
    uint64_t u(i);
    return int64_t(
                   ( (u & 0xFF00000000000000ULL) >> 56)
                  | ((u & 0x00FF000000000000ULL) >> 40)
                  | ((u & 0x0000FF0000000000ULL) >> 24)
                  | ((u & 0x000000FF00000000ULL) >> 8)
                  | ((u & 0x00000000FF000000ULL) << 8) 
                  | ((u & 0x0000000000FF0000ULL) << 24)
                  | ((u & 0x000000000000FF00ULL) << 40) 
                  | ((u & 0x00000000000000FFULL) << 56)
                   );
#  endif
#else
    uint64_t u(i);
    return int64_t(
                   ( (u & 0xFF00000000000000ULL) >> 56)
                  | ((u & 0x00FF000000000000ULL) >> 40)
                  | ((u & 0x0000FF0000000000ULL) >> 24)
                  | ((u & 0x000000FF00000000ULL) >> 8)
                  | ((u & 0x00000000FF000000ULL) << 8) 
                  | ((u & 0x0000000000FF0000ULL) << 24)
                  | ((u & 0x000000000000FF00ULL) << 40) 
                  | ((u & 0x00000000000000FFULL) << 56)
                   );
#endif
}
