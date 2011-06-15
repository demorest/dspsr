
noinst_LTLIBRARIES = libPlotting.la

if HAVE_PGPLOT

nobase_include_HEADERS = dsp/Plot.h dsp/HistoPlot.h dsp/FrequencyVsTime.h \
						 dsp/PlotFactory.h

libPlotting_la_SOURCES = Plot.C HistoPlot.C FrequencyVsTime.C PlotFactory.C

endif

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PGPLOT_CFLAGS@

