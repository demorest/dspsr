AC_DEFUN([SWIN_FUNC_GETOPT_LONG],
[
  AC_PROVIDE([SWIN_FUNC_GETOPT_LONG])

  AC_CHECK_HEADERS([getopt.h])
  AC_CHECK_FUNCS([getopt_long],have_getopt_long=yes,have_getopt_long=no)
  AM_CONDITIONAL(HAVE_GETOPT_LONG,[test x"$have_getopt_long" = xyes])

])

