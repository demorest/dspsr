
SUBDIRS	= base backends dsp psr

libs: install-binSCRIPTS
	$(MAKE) RECURSIVE_TARGETS="all-recursive libs-recursive" libs-recursive

all: config

config := $(srcdir)/config/Makefile.local

config : $(config)

#
# Convert automake variables into Standard Makefile Scheme variables
#
$(config): $(srcdir)/Makefile.am
	@echo "prefix := $(prefix)" > $@
	@echo "CC := $(CC)" >> $@
	@echo "CCC := $(CXX)" >> $@
	@echo "FC := $(F77)"  >> $@
	@echo "LF77 := @FLIBS@" >> $@
	$(pgplot.config)

if HAVE_PGPLOT
define pgplot.config 
@echo "has_PGPLOT := yes" >> $@
@echo "LCPGPLOT := @PGPLOT_LIBS@" >> $@
@echo "CPPFLAGS += @PGPLOT_CFLAGS@" >> $@
endef
endif

