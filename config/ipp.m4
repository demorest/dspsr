dnl @synopsis SWIN_LIB_IPP
dnl 
AC_DEFUN([SWIN_LIB_IPP],
[
  AC_PROVIDE([SWIN_LIB_IPP])
  AC_MSG_CHECKING([for IPP installation])

  IPP_LIBS=""
  IPP_CFLAGS=""

  ac_save_LIBS="$LIBS"
  ac_save_CPPFLAGS="$CPPFLAGS"

  ac_test_LIBS="-L/usr/local/intel/ipp41/ia32_itanium/sharedlib -lipps"
  ac_test_CPPFLAGS="-I/usr/local/intel/ipp41/ia32_itanium/include"

  LIBS="$ac_save_LIBS $ac_test_LIBS"
  CPPFLAGS="$ac_save_CPPFLAGS $ac_test_CPPFLAGS"

  AC_TRY_LINK([#include <ipps.h>],
              [ippsMalloc_32f(1);],
              have_ipp=yes, have_ipp=no)
  if test x"$have_ipp" = xyes; then
    IPP_LIBS="$ac_test_LIBS"
    AC_DEFINE(HAVE_IPP,1,[Define if IPP library is installed])
    IPP_CFLAGS="$ac_test_CPPFLAGS"
  fi

  LIBS="$ac_save_LIBS"
  CPPFLAGS="$ac_save_CPPFLAGS"

  AC_MSG_RESULT([$have_ipp])

  AC_SUBST(IPP_CFLAGS)
  AC_SUBST(IPP_LIBS)
])

