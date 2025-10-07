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
#include <sys/time.h>                

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
unsigned long page_postfetch(fault_t * f, FeatureVector *features, 
                                int *responses, int chan_id) {
    // TODO(shaurp): Confirm the accuracy of the following calculations.
#ifdef benchmark_model
    struct timeval t1, t2;
    double elapsed_time;
#endif
    uint64_t *ptr = f->page;
    void *local_addr = bkend_buf_alloc();
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
#ifdef benchmark_model
    gettimeofday(&t1, NULL);
#endif
    page_postfetch_preds(features, responses);
#ifdef benchmark_model
    gettimeofday(&t2, NULL);
	elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000000;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec);   // us to ms
    printf("%f us.\n", elapsed_time);
#endif
    /* 
     * 1. Generate address value.
     * 2. Do the same checks as the ones in readahead plus walking
     * the page table.
     * If is_page_prefetchable succeeds, the page is locked.
     * Unlocking needs to be managed by this function.
     * 3. Call local post read on the address after making a fault?
     * Or decide on a design for local post read.
     * 4. Call fault_read_done for the page.
     */
    for (int i = 0; i < 512; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val = *((uint64_t *) f->page + i); 
        if(is_page_prefetchable(f, ptr_val)) {
            fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
            /* Copy the page into the local buffer from remote */
            if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                //XXX(shaurp): Remove the pgflag ongoing flag here
                goto out; 
            }
            prefetch_read_done(ptr_val, local_addr, f);
        }
    }

    for (int i = 0; i < 512; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val =  f->page + (i - 512); 
        if(is_page_prefetchable(f, ptr_val)) { 
            fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
            /* Copy the page into a local buffer from remote */
            if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                //XXX(shaurp): Remove the pgflag ongoing flag here
                goto out; 
            }
            prefetch_read_done(ptr_val, local_addr, f);
        }
    }
   
    /*
     * 5. Unmap the pages from the kernel, mark the page as prefetched
     * in our data structures (Complete after the entire workflow is done).
     * 6. Calculate/update nevict for the prefetched pages. 
     * 7. clear the pages after fetching is done.
     */ 

out: 
    bkend_buf_free(local_addr);
    return 0; 
}
