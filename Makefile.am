
ACLOCAL_AMFLAGS = -I config

SUBDIRS	= util base backends dsp psr Management

libs:
	$(MAKE) RECURSIVE_TARGETS="all-recursive libs-recursive" libs-recursive

all: $(config)

config := $(srcdir)/config/Makefile.local

#
# Convert automake variables into Standard Makefile Scheme variables
#
$(config): $(srcdir)/Makefile.am ./Makefile
	@echo "prefix := $(prefix)" > $@
	@echo "host := $(host)" >> $@
	@echo "CC := $(CC)" >> $@
	@echo "CFLAGS := $(CFLAGS)" >> $@
	@echo "CCC := $(CXX)" >> $@
	@echo "CCFLAGS := $(CXXFLAGS)" >> $@
	@echo "FC := $(F77)"  >> $@
	@echo "FFLAGS := $(FFLAGS)" >> $@
	@echo "LF77 := @FLIBS@" >> $@
	@echo "LDFLAGS := $(LDFLAGS)" >> $@
	@echo "CPPFLAGS := $(DEFS) $(CPPFLAGS)" >> $@
	@echo "LPSRCHIVE := @PSRCHIVE_LIBS@" >> $@
	@echo "INCFLAGS += @PSRCHIVE_CFLAGS@" >> $@
	$(pgplot.config)

if HAVE_PGPLOT
define pgplot.config 
@echo "has_PGPLOT := yes" >> $@
@echo "LCPGPLOT := @PGPLOT_LIBS@" >> $@
@echo "INCFLAGS += @PGPLOT_CFLAGS@" >> $@
endef
endif

