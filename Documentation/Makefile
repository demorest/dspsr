
default:
	@if test ! -d html; then mkdir -p html; fi
	@cp `psrchive --html`/*.tag html/ || echo PSRCHIVE tags not available
	doxygen dspsr.dox > doxygen.out 2> doxygen.err
	@grep '^Error' doxygen.err || echo 2> /dev/null

install:
	./upload.csh

