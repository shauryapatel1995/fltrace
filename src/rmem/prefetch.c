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
#include "rmem/page.h"
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
                                int *responses, int chan_id, int *nevicts_needed)
{
    // TODO(shaurp): Confirm the accuracy of the following calculations.
#ifdef benchmark_model
    struct timeval t1, t2;
    double elapsed_time;
#endif
    void *local_addr = bkend_buf_alloc();
    pgthread_t owner_kthr;
    pgflags_t oldflags;
    int num_prefetches = 0, n_retries, nchunks, noverflow;
    enum fault_status status;
    unsigned long long pressure;
    uint64_t *ptr = f->page;
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
        feature->delta = i - 511; 
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
     * 7. clear the pages after fetching is done.
     */
    for (int i = 0; i < 512; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val = *((uint64_t *) f->page + i); 
        ptr_val = ptr_val & ~CHUNK_MASK;
        if(is_page_prefetchable(f, ptr_val)) {
            fprintf(stdout, "Pointer prefetch address: %lu\n", ptr_val);
            /* Copy the page into the local buffer from remote */
            if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                //XXX(shaurp): Remove the pgflag ongoing flag here
                goto out; 
            }
            prefetch_read_done(ptr_val, local_addr, f);
            fprintf(stderr, "Prefetch\n");
            RSTAT(PREFETCHES)++;
            num_prefetches++;
            clear_page_flags_and_thread(f->mr, ptr_val, 
                PFLAG_WORK_ONGOING, &oldflags, &owner_kthr);
            
            /* check that the page was locked and that the saved kthread thread id
            * (if this fault was originally from a kthread) is same or different 
            * the from current kthread id depending on if the fault was stolen */
            assert(!!(oldflags & PFLAG_WORK_ONGOING));
            if (owner_kthr) {
                assert(f->stolen_from_cq || owner_kthr == current_kthread_id);
                assert(!f->stolen_from_cq || owner_kthr != current_kthread_id);
            }

        }
    }

    for (int i = 512; i < 600; i++) {
	    if (responses[i] == 0)
		    continue;
        uint64_t ptr_val =  f->page + (i - 511); 
        ptr_val = ptr_val & ~CHUNK_MASK;
        if(is_page_prefetchable(f, ptr_val)) { 
            fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
            /* Copy the page into a local buffer from remote */
            if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                //XXX(shaurp): Remove the pgflag ongoing flag here
                goto out; 
            }
            prefetch_read_done(ptr_val, local_addr, f);
            RSTAT(PREFETCHES)++;
            fprintf(stderr, "Prefetch\n");
            num_prefetches++;
            clear_page_flags_and_thread(f->mr, ptr_val, 
                PFLAG_WORK_ONGOING, &oldflags, &owner_kthr);
            /* check that the page was locked and that the saved kthread thread id
            * (if this fault was originally from a kthread) is same or different 
            * the from current kthread id depending on if the fault was stolen */
            assert(!!(oldflags & PFLAG_WORK_ONGOING));
            if (owner_kthr) {
                assert(f->stolen_from_cq || owner_kthr == current_kthread_id);
                assert(!f->stolen_from_cq || owner_kthr != current_kthread_id);
            }

        }
    }
   
    /*
     * 5. Unmap the pages from the kernel, mark the page as prefetched
     * in our data structures (Complete after the entire workflow is done).
     * 6. Calculate/update nevict for the prefetched pages. 
     */ 

    assert(nevicts_needed);
    /* book some memory for the pages */
    if (num_prefetches > 0) {
        assert(num_prefetches > 0);
        pressure = atomic64_add_and_fetch(&memory_used, num_prefetches * CHUNK_SIZE);
        log_debug("%s - memory pressure during prefetch %llu, limit %lu", FSTR(f), 
            pressure, local_memory);
        if (pressure > local_memory) {
            noverflow = (pressure - local_memory) / CHUNK_SIZE;
            *nevicts_needed = (noverflow < nchunks) ? noverflow : nchunks;
        }

        /* update maximum memory usage counter. FIXME: should use CAS! */
        if (pressure > atomic64_read(&max_memory_used))
            atomic64_write(&max_memory_used, pressure);

        log_debug("%s - %d page(s) prefetched with return status %d, pressure %llu"
            " evicts %d", FSTR(f), nchunks, status, pressure, *nevicts_needed);
    }
out: 
    bkend_buf_free(local_addr);
    return 0; 
}
