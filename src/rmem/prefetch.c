/*
 * prefetch.c - Prefetcher backend implementation
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <dlfcn.h>
#include "rmem/common.h"
#include "rmem/prefetch.h"
/*
 * Implementation to do the actual prefetching with predictions from a
 * prefetching policy. The function does the following - 
 * 1. Constructs the required input for a prefetching policy.
 * 2. Obtains predictions from a given policy.
 * 3. Converts predictions to addresses to prefetch
 * 4. Checks whether the page for an address can be prefetched.
 * 5. Prefetches the page, and updates the eviction count to reflect 
 * the additional pages.
*/
unsigned long page_postfetch(fault_t * f, FeatureVector *features, int *responses) {
    // TODO(shaurp): Confirm the accuracy of the following calculations.
    uint64_t *ptr = f->page;
    int num_prefetches = 0;
    int faulting_location = (f->faulting_addr - f->page) / sizeof(uint64_t);
    /* Setup pointer features */
    for(int i = 0; i < 512; i++, ptr++) {
	FeatureVector *feature = &features[i];
	feature->pc = f->pc;
	feature->offset = i;
	feature->delta = (*ptr - f->page) / 4096;
	feature->offset_from_faulting = i - faulting_location;
    }
    /* Setup next-N features */
    for (int i = 512; i < 600; i++) {
	FeatureVector *feature = &features[i];
	feature->pc = f->pc;
	feature->offset = 0; 
	feature->delta = i - 512; 
	feature->offset_from_faulting = 0;
    } 
    page_postfetch_preds(features, responses);
    
    /* 
     * 1. Generate address value.
     * 2. Do the same checks as the ones in readahead plus walking
     * the page table. 
     */
    for (int i = 0; i < 512; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val = *((uint64_t *) f->page + i); 
        if(is_page_prefetchable(f, ptr_val))
            fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
    }

    for (int i = 0; i < 512; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val =  f->page + (i - 512); 
        if(is_page_prefetchable(f, ptr_val))
            fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
    }
    /*
     * 3. Call local post read on the address after making a fault?
     * Or decide on a design for local post read.
     */

    /*
     * 4. Call fault_read_done for the page.
     * 5. Calculate/update nevict for the prefetched pages. 
     * 6. clear the pages after fetching is done.
     */ 
    assert(num_prefetches > 0);
    
}
