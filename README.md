# tiny_cache
A light Python simulator to validate and compare ideal Cache Replacement Policy performance.

Major Functions:
1. *run_sim(cache_size, asize, bsize, repl_policy)*: Runs the sim NUM_ACCESS times with array sizes asize and bsize.
2. *alternate_access_pattern()*: An example access pattern generator that alternates between multiple arrays in a round-robin fashion.
3. *cache_access()*: Handles Cache Access leading to a hit or a miss.
4. *update_statistics()*: Computes the Replacement Policy related metrics. Provided functions handles eva, lru and random.
