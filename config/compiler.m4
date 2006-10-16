dnl @synopsis SWIN_COMPILER
dnl 
AC_DEFUN([SWIN_COMPILER],
[
  AC_PROVIDE([SWIN_COMPILER])

  AC_ARG_WITH([compiler],
              AC_HELP_STRING([--with-compiler=PKG],
                             [PKG=intel,portland,gnu,gcc-4]))

  # "yes" is not a specification
  if test x"$with_compiler" = xyes; then
    with_compiler=
  fi

  # Portland Group
  if test x"$with_compiler" = xportland; then
    AC_MSG_NOTICE([Setting environment variables for Portland Group compiler])
    CXX=pgCC
    CC=pgcc
    F77=pgf77
    CXXFLAGS="--instantiate=local"
  fi

  # GNU Compiler Collection version 4 on Mac
  if test x"$with_compiler" = xgcc-4; then
    AC_MSG_NOTICE([Setting environment variables for Mac GCC 4])
    CXX=g++-4
    CC=gcc-4
    F77=gfortran
  fi

])

