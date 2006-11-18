dnl @synopsis SWIN_OPTIONS_SET
dnl 
AC_DEFUN([SWIN_OPTIONS_SET],
[
  AC_PROVIDE([SWIN_OPTIONS_SET])

  if test "$CXXFLAGS"; then
    swin_cxxflags_set="yes"
  fi

  if test "$CFLAGS"; then
    swin_cflags_set="yes"
  fi

  if test "$FFLAGS"; then
    swin_fflags_set="yes"
  fi

])

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

    if test x"$swin_cxxflags_set" != xyes; then
      CXXFLAGS=`echo $CXXFLAGS | sed -e 's/-g\>//g'`
    else
      AC_MSG_NOTICE([   CXXFLAGS set by user.])
    fi

    if test x"$swin_cflags_set" != xyes; then
      CFLAGS=`echo $CFLAGS | sed -e 's/-g\>//g'`
    else
      AC_MSG_NOTICE([   CFLAGS set by user.])
    fi

    if test x"$swin_fflags_set" != xyes; then
      FFLAGS=`echo $FFLAGS | sed -e 's/-g\>//g'`
    else
      AC_MSG_NOTICE([   FFLAGS set by user.])
    fi

  else
    AC_MSG_WARN([Debugging information not disabled.  Binaries may be large.])
  fi

])

