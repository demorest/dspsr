
lib_LTLIBRARIES = libdspstats.la

libdspstats_la_SOURCES = PearsonIV.C SKLimits.C

nobase_include_HEADERS = dsp/MidPoint.h dsp/Trapezoid.h \
	dsp/Romberg.h dsp/Neville.h dsp/VolumeIntegral.h \
	dsp/PearsonIV.h dsp/SKLimits.h dsp/NewtonRaphson.h

bin_PROGRAMS = test_Romberg sklimit

test_Romberg_SOURCES = test_Romberg.C 

sklimit_SOURCES = sklimit.C

if HAVE_PGPLOT

  ###########################################################################
  #
  # start PGPLOT-specific code
  #

  bin_PROGRAMS += sk_distrib_plot sk_limit_plot

  sk_distrib_plot_SOURCES = sk_distrib_plot.C
  sk_limit_plot_SOURCES = sk_limit_plot.C

  #
  # end PGPLOT-specific code
  #
  ###########################################################################

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libdspstats.la $(top_builddir)/Kernel/libdspbase.la \
	@PGPLOT_LIBS@ -lm


AM_CPPFLAGS += @PGPLOT_CFLAGS@


