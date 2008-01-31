# SWIN_LIB_PSRDADA([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_PSRDADA],
[
  AC_PROVIDE([SWIN_LIB_PSRDADA])

  AC_ARG_WITH([psrdada-dir],
              AC_HELP_STRING([--with-psrdada-dir=DIR],
                             [PSRDADA is installed in DIR]))

  PSRDADA_CFLAGS=""
  PSRDADA_LIBS=""

  if test x"$with_psrdada_dir" = xno; then
    # user disabled psrdada. Leave cache alone.
    have_psrdada="User disabled PSRDADA."
  else

    AC_MSG_CHECKING([for PSRDADA installation])

    # "yes" is not a specification
    if test x"$with_psrdada_dir" = xyes; then
      with_psrdada_dir=
    fi

    if test x"$with_psrdada_dir" = x; then
      psrdada_cflags=`which psrdada_cflags`
      psrdada_ldflags=`which psrdada_ldflags`
    else
      psrdada_cflags=$with_psrdada_dir/bin/psrdada_cflags
      psrdada_ldflags=$with_psrdada_dir/bin/psrdada_ldflags
    fi

    have_psrdada="not found"

    if test -x "$psrdada_cflags" -a -x "$psrdada_ldflags" ; then

      ac_save_CPPFLAGS="$CPPFLAGS"
      ac_save_LIBS="$LIBS"

      CPPFLAGS="`$psrdada_cflags` $CPPFLAGS"
      LIBS="`$psrdada_ldflags` $LIBS"

      AC_TRY_LINK([#include "dada_hdu.h"], [dada_hdu_create (0);],
                  have_psrdada=yes, have_psrdada=no)

      if test $have_psrdada = yes; then
        PSRDADA_CFLAGS="`$psrdada_cflags`"
        PSRDADA_LIBS="`$psrdada_ldflags`"
      fi

      LIBS="$ac_save_LIBS"
      CPPFLAGS="$ac_save_CPPFLAGS"

    fi

  fi

  AC_MSG_RESULT([$have_psrdada])

  if test "$have_psrdada" = "yes"; then
    AC_DEFINE([HAVE_PSRDADA], [1], [Define if the PSRDADA library is present])
    [$1]
  else
    if test "$have_psrdada" = "not found"; then
      echo
      AC_MSG_NOTICE([Ensure that PSRDADA executables are in PATH.])
      AC_MSG_NOTICE([Alternatively, use the --with-psrdada-dir option.])
      echo
    fi
    [$2]
  fi

  AC_SUBST(PSRDADA_LIBS)
  AC_SUBST(PSRDADA_CFLAGS)
  AM_CONDITIONAL(HAVE_PSRDADA,[test "$have_psrdada" = "yes"])

])

