dnl @synopsis SWIN_LIB_MPI
dnl 
AC_DEFUN([SWIN_LIB_MPI],
[
  AC_PROVIDE([SWIN_LIB_MPI])

  AC_ARG_WITH([mpi],
              AC_HELP_STRING([--with-mpi],
                             [Compile Message Passing Interface (MPI) codes]))
  AC_ARG_WITH([mpi-dir],
              AC_HELP_STRING([--with-mpi-dir=DIR],
                             [MPI is in DIR]))

  AC_ARG_WITH([mpi-inc-dir],
              AC_HELP_STRING([--with-mpi-inc-dir=DIR],
                             [MPI header files are in DIR]))

  AC_ARG_WITH([mpi-lib-dir],
              AC_HELP_STRING([--with-mpi-lib-dir=DIR],
                             [MPI library is in DIR]))

  AC_ARG_WITH([mpi-link],
              AC_HELP_STRING([--with-mpi-link=ARGS],
                             [MPI link arguments]))

  MPI_CFLAGS=""
  MPI_LIBS=""

  if test x"$with_mpi" != x"yes"; then

    AC_MSG_NOTICE([MPI not enabled.])
    have_mpi=no

  else

    # "yes" is not a specification
    if test x"$with_mpi_dir" = xyes; then
      with_mpi_dir=
    fi
    if test x"$with_mpi_inc_dir" = xyes; then
      if test x"$with_mpi_dir" = xyes; then
        with_mpi_inc_dir=$with_mpi_dir/include
      else
        with_mpi_inc_dir=
      fi
    fi
    if test x"$with_mpi_lib_dir" = xyes; then
      if test x"$with_mpi_dir" = xyes; then
        with_mpi_lib_dir=$with_mpi_dir/lib
      else
        with_mpi_lib_dir=
      fi
    fi
    if test x"$with_mpi_link" = xyes; then
      with_mpi_link=
    fi

    AC_MSG_CHECKING([for MPI installation])

    ## Look for the header file ##
    cf_include_path_list="$with_mpi_inc_dir .
                          $PSRHOME/packages/$LOGIN_ARCH/lam/include
                          $PSRHOME/packages/$LOGIN_ARCH/mpich/include
                          /usr/local/include/mpi
                          /usr/local/mpi/include"

    ac_save_CPPFLAGS="$CPPFLAGS"

    AC_LANG_PUSH(C++)

    for cf_dir in $cf_include_path_list; do
      CPPFLAGS="-I$cf_dir $ac_save_CPPFLAGS"
      AC_TRY_COMPILE([#include <mpi.h>],[MPI_Init(0,0);],
                     have_mpi=yes, have_mpi=no)

      if test x"$have_mpi" = xyes; then
        if test x"$cf_dir" = x.; then
          MPI_CFLAGS=""
        else
          MPI_CFLAGS="-I$cf_dir"
        fi
        break
      fi
    done

    AC_LANG_POP(C++)

    if test x"$have_mpi" = xyes; then

      ## Look for the library ##
      cf_lib_path_list="$with_mpi_lib_dir .
                        $PSRHOME/packages/$LOGIN_ARCH/lam/lib
                        $PSRHOME/packages/$LOGIN_ARCH/mpich/lib
                        /usr/local/lib
                        /usr/local/mpi/lib"

      ac_save_LIBS="$LIBS"

      if test x"$with_mpi_link" != x; then

        LIBS="$with_mpi_link"
        AC_TRY_LINK([#include <mpi.h>],[MPI_Init(0,0);],
                    have_mpi=yes, have_mpi=no)

        if test $have_mpi = yes; then
          MPI_LIBS="$LIBS"
        else
          AC_MSG_RESULT(no)
          AC_MSG_ERROR([User specified MPI link arguments failed])
        fi

      fi

      for cf_dir in $cf_lib_path_list; do

        if test x"$MPI_LIBS" != x; then
          break
        fi

        LIBS="-L$cf_dir -lmpi -llam -lpthread $ac_save_LIBS"

        AC_TRY_LINK([#include <mpi.h>],[MPI_Init(0,0);],
                    have_mpi=lam, have_mpi=no)
        if test x"$have_mpi" != xno; then
          if test x"$cf_dir" = x.; then
            MPI_LIBS="-lmpi -llam -lpthread"
          else
            MPI_LIBS="-L$cf_dir -lmpi -llam -lpthread"
          fi
          break
        fi


        LIBS="-L$cf_dir -lmpich $ac_save_LIBS"
        AC_TRY_LINK([#include <mpi.h>],[MPI_Init(0,0);],
                    have_mpi=mpich, have_mpi=no)
        if test x"$have_mpi" != xno; then
          if test x"$cf_dir" = x.; then
            MPI_LIBS="-lmpich"
          else
            MPI_LIBS="-L$cf_dir -lmpich"
          fi
          break
        fi

      done

      LIBS="$ac_save_LIBS"

    fi

    CPPFLAGS="$ac_save_CPPFLAGS"

    AC_MSG_RESULT([$have_mpi])

  fi

  if test x"$have_mpi" != xno; then
    AC_DEFINE([HAVE_MPI],[1],
              [Define if a Message Passing Interface library is present])
    [$1]
  else
    AC_MSG_WARN([Message Passing Interface (MPI) code will not be compiled])
    [$2]
  fi

  AC_SUBST(MPI_LIBS)
  AC_SUBST(MPI_CFLAGS)
  AM_CONDITIONAL(HAVE_MPI,[test x"$have_mpi" != xno])

])

