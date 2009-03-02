#
# SWIN_LIB_PRESTO([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
#
# This m4 macro checks availability of the PRESTO Library by Scott Ransom 
#
# PRESTO_CFLAGS - autoconfig variable with flags required for compiling
# PRESTO_LIBS   - autoconfig variable with flags required for linking
# HAVE_PRESTO   - automake conditional
# HAVE_PRESTO   - pre-processor macro in config.h
#
# This macro tries to link a test program, first using only 
#
#    -L$PRESTO -lpresto
#
# Notice that the environment variable PRESTO is used.
#
#
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_PRESTO],
[
  AC_PROVIDE([SWIN_LIB_PRESTO])

  AC_MSG_CHECKING([for PRESTO installation])

  PRESTO_CFLAGS=""
  PRESTO_LIBS=""

  if test x"$PRESTO" != x; then
    PRESTO_CFLAGS="-I$PRESTO/include"
    PRESTO_LIBS="-L$PRESTO/lib"
  fi

  PRESTO_LIBS="$PRESTO_LIBS -lpresto"

  ac_save_CFLAGS="$CFLAGS"
  ac_save_LIBS="$LIBS"
  LIBS="$ac_save_LIBS $PRESTO_LIBS"
  CFLAGS="$ac_save_CFLAGS $PRESTO_CFLAGS"

  AC_TRY_LINK([#include <makeinf.h>],[infodata* data=0; readinf(data,0);],
              have_presto=yes, have_presto=no)

  AC_MSG_RESULT($have_presto)

  LIBS="$ac_save_LIBS"
  CFLAGS="$ac_save_CFLAGS"

  if test x"$have_presto" = xyes; then
    AC_DEFINE([HAVE_PRESTO], [1], [Define to 1 if you have the PRESTO library])
    [$1]
  else
    AC_MSG_WARN([PRESTO code will not be compiled])
    if test x"$PRESTO" = x; then
      AC_MSG_WARN([Please set the PRESTO environment variable])
    fi
    PRESTO_CFLAGS=""
    PRESTO_LIBS=""
    [$2]
  fi

  AC_SUBST(PRESTO_CFLAGS)
  AC_SUBST(PRESTO_LIBS)
  AM_CONDITIONAL(HAVE_PRESTO, [test x"$have_presto" = xyes])

])

