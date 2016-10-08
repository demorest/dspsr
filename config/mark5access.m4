# SWIN_LIB_MARK5ACCESS([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_MARK5ACCESS],
[
  AC_PROVIDE([SWIN_LIB_MARK5ACCESS])

  AC_ARG_WITH([mark5access-dir],
              AC_HELP_STRING([--with-mark5access-dir=DIR],
                             [MARK5ACCESS is installed in DIR]))

  MARK5ACCESS_CFLAGS=""
  MARK5ACCESS_LIBS=""

  if test x"$with_mark5access_dir" = xno; then
    # user disabled mark5access. Leave cache alone.
    have_mark5access="User disabled mark5access."
  else

    AC_MSG_CHECKING([for mark5access installation])

    # "yes" is not a specification
    if test x"$with_mark5access_dir" = xyes; then
      with_mark5access_dir=
    fi

    have_mark5access="not found"

    ac_save_CPPFLAGS="$CPPFLAGS"
    ac_save_LIBS="$LIBS"

    CPPFLAGS="`pkg-config --cflags mark5access` $CPPFLAGS"
    LIBS="`pkg-config --libs mark5access` $LIBS"

    # TESTPKG="`pkg-config --cflags mark5access`"
    # AC_MSG_NOTICE([pkg-config returns $TESTPKG])
    AC_TRY_LINK([#include <mark5access.h>], [new_mark5_stream(0,0);],
                have_mark5access=yes, have_mark5access=no)

    if test $have_mark5access = yes; then
      MARK5ACCESS_CFLAGS="`pkg-config --cflags mark5access`"
      MARK5ACCESS_LIBS="`pkg-config --libs mark5access`"
    fi

    LIBS="$ac_save_LIBS"
    CPPFLAGS="$ac_save_CPPFLAGS"

  fi

  AC_MSG_RESULT([$have_mark5access])

  if test "$have_mark5access" = "yes"; then
    AC_DEFINE([HAVE_MARK5ACCESS], [1], [Define if the mark5access library is present])
    [$1]
  else
    AC_MSG_NOTICE([Ensure that the PKG_CONFIG_PATH environment variable points to])
    AC_MSG_NOTICE([the lib/pkgconfig sub-directory of the root directory where])
    AC_MSG_NOTICE([the mark5access library was installed.])
    AC_MSG_NOTICE([Alternatively, use the --with-mark5access-dir option.])
    [$2]
  fi

  AC_SUBST(MARK5ACCESS_LIBS)
  AC_SUBST(MARK5ACCESS_CFLAGS)
  AM_CONDITIONAL(HAVE_MARK5ACCESS,[test "$have_mark5access" = "yes"])

])

