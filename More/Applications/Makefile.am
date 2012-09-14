if HAVE_PGPLOT

bin_PROGRAMS = searchplot

searchplot_SOURCES = searchplot.C
searchplot_LDADD = $(LDADD) @PGPLOT_LIBS@

LDADD = $(top_builddir)/More/Plotting/libPlotting.la \
				$(top_builddir)/Kernel/libdspbase.la \
				$(top_builddir)/Signal/General/libdspdsp.la \
				$(top_builddir)/Signal/Statistics/libdspstats.la \
				$(top_builddir)/Kernel/Classes/libClasses.la \
				@CUFFT_LIBS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @CFITSIO_CFLAGS@ @CUFFT_CFLAGS@

endif
