/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/*
 * Macro to declare what class of machine is being compiled for.
 */

#ifndef _CLASS_H
#define _CLASS_H

#if pspm | pspm2

/*
 * Penn State Pulsar Machine (PSPM) derivative, i.e., analog filterbank.
 */
#define PSPM 1

#elif nbpp | lbpp | bacspin | ebpp

/*
 * Berkeley Pulsar Processor (BPP) derivative, i.e., digital filter bank.
 */
#define BPP 1

#elif cbr | cbrl | cpsr | tmfe

/*
 * Caltech Baseband Recorder (CBR) derivative, i.e., baseband recorder.
 */
#define CBR 1

#endif

#endif /* _CLASS_H */
