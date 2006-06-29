dnl @synopsis SWIN_LIB_FFTW
dnl 
AC_DEFUN([SWIN_LIB_FFTW],
[
  AC_PROVIDE([SWIN_LIB_FFTW])

  AC_MSG_CHECKING([for single-precision FFTW-2 header])

  AC_TRY_COMPILE([#include <fftw.h>],
                 [#ifndef FFTW_ENABLE_FLOAT
                  #error must have single-precision library
                  #endif],
                 have_fftw=yes, have_fftw=no)

  AC_MSG_RESULT($have_fftw)

  ac_save_LIBS="$LIBS"
  FFTW_LIBS=""
  FFTW_CFLAGS=""

  if test $have_fftw = yes; then

    AC_MSG_CHECKING([for FFTW-2 library])

    LIBS="-lfftw $ac_save_LIBS"
    AC_TRY_LINK([#include <fftw.h>],
                [fftw_create_plan (1024, FFTW_FORWARD, FFTW_ESTIMATE);],
                have_fftw=yes, have_fftw=no)

    AC_MSG_RESULT($have_fftw)

    if test $have_fftw = yes; then
      AC_DEFINE(HAVE_FFTW,1,[Define if the FFTW library is installed])
      FFTW_LIBS="-lfftw"
    fi

    AC_MSG_CHECKING([for FFTW-2 real-to-complex library])

    LIBS="-lrfftw -lfftw $ac_save_LIBS"
    AC_TRY_LINK([#include <fftw.h>],
                [rfftw_create_plan (1024, FFTW_FORWARD, FFTW_ESTIMATE);],
                have_rfftw=yes, have_rfftw=no)

    AC_MSG_RESULT($have_rfftw)

    if test $have_rfftw = yes; then
      AC_DEFINE(HAVE_RFFTW,1,[Define if the FFTW real library is installed])
      FFTW_LIBS="-lrfftw -lfftw"
    fi

  fi

  AC_MSG_CHECKING([for single-precision FFTW-3 library])

  LIBS="-lfftw3f $ac_save_LIBS"
  AC_TRY_LINK([#include <fftw3.h>],
              [fftwf_plan_dft_1d (1024, 0, 0, FFTW_FORWARD, FFTW_ESTIMATE);],
              have_fftw3=yes, have_fftw3=no)

  AC_MSG_RESULT($have_fftw3)

  if test $have_fftw3 = yes; then
    AC_DEFINE(HAVE_FFTW3,1,[Define if the FFTW3 library is installed])
    FFTW_LIBS="-lfftw3f $FFTW_LIBS"
  fi

  AM_CONDITIONAL(HAVE_FFTW2,[test x"$have_fftw" != xno])

  AC_SUBST(FFTW_LIBS)
  AC_SUBST(FFTW_CFLAGS)
  LIBS="$ac_save_LIBS"

])

