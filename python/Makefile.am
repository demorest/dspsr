
nodist_python_PYTHON = dspsr.py

python_LTLIBRARIES = _dspsr.la

_dspsr_la_SOURCES = dspsr.i

nodist__dspsr_la_SOURCES = dspsr_wrap.cxx

_dspsr_la_LDFLAGS = -module -avoid-version -no-undefined

_dspsr_la_LIBADD = \
		   $(top_builddir)/Signal/Pulsar/libdspsr.la \
		   $(top_builddir)/Signal/General/libdspdsp.la \
		   $(top_builddir)/Signal/Statistics/libdspstats.la \
		   $(top_builddir)/Kernel/libdspbase.la 

SWIG_FLAGS = -Wall -python -I$(top_builddir)/local_include @PSRCHIVE_INCLUDE@

dspsr_wrap.cxx: $(srcdir)/dspsr.i
	$(SWIG) $(SWIG_FLAGS) $(srcdir)/dspsr.i

CLEANFILES = dspsr.py dspsr.pyc dspsr_wrap.cxx

#############################################################################
#

INCLUDES = @SWIG_PYTHON_CPPFLAGS@ -I$(top_builddir)/local_include @PSRCHIVE_CFLAGS@

