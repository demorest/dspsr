
#ifndef CHARSTAR
#include "y.tab.h" /* pull in the defs for the "type" enum */
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct HEADERKEY {
  void *next;
  char *name;
  int offset;
  int type;
  int len;  /* length of data element */
  int alen; /* array length */
};

struct HEADERVAL {
  void *value;
  struct HEADERKEY *key;
};

struct HEADERP {
  struct HEADERKEY *head;
  struct HEADERKEY *tail;
  char *buf;       /* ascii C header declaration */
  int offset;      /* len of buf ( offset to start of generic head in file */
  void *header;    /* pointer to instance of generic header */
  int headlen;     /* len of generic header */
  int fd;          /* file des */
  int yacc_offset; /* last returned by head_input */
};

struct HEADERP *head_parse (const char *);

int fetch_hdrval (struct HEADERP *h, char *name, void *dest, int ndest);

int close_parse (struct HEADERP *h);

extern struct HEADERKEY headerkey[];

#ifdef __cplusplus
}
#endif
