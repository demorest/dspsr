SUBDIRS =

lib_LTLIBRARIES = libdspsrmore.la
libdspsrmore_la_SOURCES = 
libdspsrmore_la_LIBADD =

if HAVE_PGPLOT
SUBDIRS += Plotting

libdspsrmore_la_LIBADD += Plotting/libPlotting.la

SUBDIRS += Applications
endif

include $(top_srcdir)/config/Makefile.include

