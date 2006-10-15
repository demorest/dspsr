# SWIN_LIB_PSRCHIVE([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_PSRCHIVE],
[
  AC_PROVIDE([SWIN_LIB_PSRCHIVE])

  AC_ARG_WITH([psrchive-dir],
              AC_HELP_STRING([--with-psrchive-dir=DIR],
                             [PSRCHIVE is installed in DIR]))

  PSRCHIVE_CFLAGS=""
  PSRCHIVE_LIBS=""

  if test x"$with_psrchive_dir" = x"no"; then
    # user disabled psrchive. Leave cache alone.
    have_psrchive="User disabled PSRCHIVE."
  else

    AC_MSG_CHECKING([for PSRCHIVE installation])

    # "yes" is not a specification
    if test x"$with_psrchive_dir" != xyes; then
      psrchive_cflags=$with_psrchive_dir/bin/psrchive_cflags
      psrchive_ldflags=$with_psrchive_dir/bin/psrchive_ldflags
    else
      psrchive_cflags=`which psrchive_cflags`
      psrchive_ldflags=`which psrchive_ldflags`
    fi

    have_psrchive="no"

    if test -x $psrchive_cflags -a -x $psrchive_ldflags ; then

      ac_save_CPPFLAGS="$CPPFLAGS"
      ac_save_LIBS="$LIBS"

      CPPFLAGS="`$psrchive_cflags` $CPPFLAGS"
      LIBS="`$psrchive_ldflags` $LIBS"

      AC_LANG_PUSH(C++)
      AC_TRY_LINK([#include "Pulsar/Archive.h"], [Pulsar::Archive::load("");],
                  have_psrchive=yes, have_psrchive=no)
      AC_LANG_POP(C++)

      if test $have_psrchive = yes; then
        PSRCHIVE_CFLAGS="`$psrchive_cflags`"
        PSRCHIVE_LIBS="`$psrchive_ldflags`"
      fi

    fi

    LIBS="$ac_save_LIBS"
    CPPFLAGS="$ac_save_CPPFLAGS"

  fi

  AC_MSG_RESULT([$have_psrchive])

  if test $have_psrchive = yes; then
    AC_DEFINE([HAVE_PSRCHIVE], [1], [Define if the PSRCHIVE library is present])
    [$1]
  else
    AC_MSG_WARN([PSRCHIVE library not found])
    [$2]
  fi

  AC_SUBST(PSRCHIVE_LIBS)
  AC_SUBST(PSRCHIVE_CFLAGS)
  AM_CONDITIONAL(HAVE_PSRCHIVE,[test $have_psrchive = yes])

])

