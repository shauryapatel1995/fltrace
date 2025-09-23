#ifndef __PREFETCH_H__
#define __PREFETCH_H__

/* methods */
void init_prefetcher();
unsigned long page_prefetch();
unsigned long page_postfetch();

#endif // __PREFETCH_H
