dnl @synopsis SWIN_DEBUG
dnl 
AC_DEFUN([SWIN_DEBUG],
[
  AC_PROVIDE([SWIN_DEBUG])

  AC_ARG_ENABLE([debug],
                AC_HELP_STRING([--enable-debug],
                               [Enable debugging information]),
                [swin_debug=yes])

  if test x"$swin_debug" != xyes; then
    AC_MSG_NOTICE([Disabling debugging information compiler option (-g)])
    CXXFLAGS=`echo $CXXFLAGS | sed -e 's/-g//g'`
    CFLAGS=`echo $CFLAGS | sed -e 's/-g//g'`
    FFLAGS=`echo $FFLAGS | sed -e 's/-g//g'`
  else
    AC_MSG_WARN([Debugging information enabled.  Binaries will be large.])
  fi

])

