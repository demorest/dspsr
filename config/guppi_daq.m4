#
# SWIN_LIB_GUPPI_DAQ([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
#
# This checks for guppi_daq to be installed.  The GUPPI_DIR env 
# variable must be set.
#
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_GUPPI_DAQ],
[
  AC_PROVIDE([SWIN_LIB_GUPPI_DAQ])

  AC_MSG_CHECKING([for guppi_daq installation])

  GUPPI_DAQ_CFLAGS=""
  GUPPI_DAQ_LIBS=""
  have_guppi_daq="no"

  if test x"$GUPPI_DIR" != x; then
    GUPPI_DAQ_CFLAGS="-I$GUPPI_DIR/src"
    GUPPI_DAQ_LIBS="-L$GUPPI_DIR/src -lguppi_daq -lsla -lvdifio -lm"
    have_guppi_daq="yes"
  fi

  AC_MSG_RESULT($have_guppi_daq)

  if test x"$have_guppi_daq" = xyes; then
    AC_DEFINE([HAVE_GUPPI_DAQ], [1], [Define to 1 if you have the GUPPI_DAQ library])
    [$1]
  else
    AC_MSG_WARN([guppi_daq code will not be compiled])
    if test x"$GUPPI_DIR" = x; then
      AC_MSG_WARN([Please set the GUPPI_DIR environment variable])
    fi
    GUPPI_DAQ_CFLAGS=""
    GUPPI_DAQ_LIBS=""
    [$2]
  fi

  AC_SUBST(GUPPI_DAQ_CFLAGS)
  AC_SUBST(GUPPI_DAQ_LIBS)
  AM_CONDITIONAL(HAVE_GUPPI_DAQ, [test x"$have_guppi_daq" = xyes])

])

