from collections import deque
import numpy as np

class CacheLine:
    def __init__(self, addr):
        self.addr = addr
        self.age = 0

class HexDeque(deque):
    def __str__(self):
        return "[" + ", ".join(hex(item) for item in self) + "]"

#VERBOSE > 0 : Shows each cache line access with hit and miss info.
VERBOSE = 1

CACHE_SIZE = 6
cache = []
fifo_queue = HexDeque()

# Initialize hit and miss counters for each age 
hit_counters = {}
miss_counters = {}
eviction_counters = {}

# Initialize cumulative hit, miss, lifetime counters
lifetimes_a = np.array([])
hits_a = np.array([])
miss_a = np.array([])
evictions_a = np.array([])

hits_gt_a = np.array([])
evictions_gt_a = np.array([])
lifetimes_gt_a = np.array([])
expected_lifetimes_a = np.array([])
EVA = np.array([])

# Arrays A and B information
array_a = {"start_addr": 0x100, "length": 2}
array_b = {"start_addr": 0x200, "length": 6}
a_arr = np.arange(array_a["start_addr"], array_a["start_addr"]+array_a["length"])
b_arr = np.arange(array_b["start_addr"], array_b["start_addr"]+array_b["length"])
A_hits = 0
B_hits = 0

current_array = None  # Keep track of the currently accessed array
array_index_a = -1  # Index to keep track of the current position in array A
array_index_b = -1  # Index to keep track of the current position in array B
NUM_ACCESS = 40  # Total number of accesses

def alternate_access_pattern():
    global current_array, array_index_a, array_index_b

    if current_array is None or current_array == array_b:
        current_array = array_a  # Switch to array A
        array_index_a = (array_index_a + 1) % array_a["length"]  # Wrap around for array A
    else:
        current_array = array_b  # Switch to array B
        array_index_b = (array_index_b + 1) % array_b["length"]  # Wrap around for array B

    if current_array == array_a:
        addr = array_a["start_addr"] + array_index_a
    else:
        addr = array_b["start_addr"] + array_index_b

    return addr

def cache_access(addr):
    global A_hits, B_hits
    hit = 0
        
    if VERBOSE>0:
        print(f"Accessing:  0x{addr:04X}")
    index = find_cache_line(addr)
    # Increment the ages of all cache lines
    increment_cache_ages() #TODO: At beginning or end??

    if index is not None:
        # Cache hit
        hit = 1
        update_hit_counters(cache[index].age)
        # Remove cache element, MRU addition at end.
        fifo_queue.remove(cache[index].addr)
        cache[index].age = 0  # Set the age of the accessed cache line to 0
        if VERBOSE>0:
            print(f"Cache hit: Address 0x{addr:04X} found in cache at index {index}")
    else:
        if len(cache) >= CACHE_SIZE:
            index_to_replace = find_cache_line(fifo_queue.popleft())
            replace_cacheline(index_to_replace, addr)
        else:
            cache_line = CacheLine(addr)
            cache.append(cache_line)
            #cache[-1].age = 0  # Implicit # Set the age of the newly added cache line to 0
            #UPDATING MISS COUNTER IN COMPULSORY MISS
            update_miss_counters(cache_line.age, compulsory=1)
            if VERBOSE>0:
                print(f"Cache miss: Address 0x{addr:04X} added to cache")
    
    if (addr in a_arr):
        A_hits += hit
    if (addr in b_arr):
        B_hits += hit
    
    fifo_queue.append(addr)
    return hit

def find_cache_line(addr):
    for i, cache_line in enumerate(cache):
        if cache_line.addr == addr:
            return i
    return None

def replace_cacheline(index, new_addr):
    old_age = cache[index].age
    old_addr = cache[index].addr
    update_miss_counters(old_age, compulsory=0)

    # Replace the cache line
    if VERBOSE>0:
        print(f"Cache miss: Replacing cache line at index {index} with Address 0x{new_addr:04X}. Old Address 0x{old_addr:04X} is evicted.")
        #print("Q contents:", fifo_queue, "\n")
    cache[index] = CacheLine(new_addr)

def increment_cache_ages():
    for cache_line in cache:
        cache_line.age += 1

def update_hit_counters(age):
    if age not in hit_counters:
        hit_counters[age] = 1
    else:
        hit_counters[age] += 1

def update_miss_counters(age, compulsory):
    if age not in miss_counters: #miss==evictions except for compulsory misses
        miss_counters[age] = 1
    else:
        miss_counters[age] += 1
    if not compulsory:
        if age not in eviction_counters:
            eviction_counters[age] = 1
        else:
            eviction_counters[age] += 1

def print_cache_contents():
    print("Current cache contents:")
    print("*****")
    for i, cache_line in enumerate(cache):
        print(f"Cache Line {i}: Address 0x{cache_line.addr:04X}, Age {cache_line.age}")

def upscan(arr):
    rev = arr[::-1]
    tot = np.cumsum(rev)
    tot_rev = tot[::-1]
    res = np.append(tot_rev[1:], 0)
    return res    
    
def update_statistics():
    global hits_a, miss_a, evictions_a, lifetimes_a, hits_gt_a, evictions_gt_a, lifetimes_gt_a
    global expected_lifetimes_a
    
    max_age = max(max(hit_counters.keys(), default=0), 
                  max(miss_counters.keys(), default=0), 
                  max(eviction_counters.keys(), default=0))
    
    for age in range(max_age,-1,-1):
        hits_a = np.insert(hits_a, 0, hit_counters.get(age, 0))
        miss_a = np.insert(miss_a, 0, miss_counters.get(age, 0))
        evictions_a = np.insert(evictions_a, 0, eviction_counters.get(age, 0))
                     
    lifetimes_a = hits_a + evictions_a
    hits_gt_a = upscan(hits_a)
    evictions_gt_a = upscan(evictions_a)
    lifetimes_gt_a = upscan(lifetimes_a) #hits_gt_a + evictions_gt_a
    expected_lifetimes_a = np.cumsum(lifetimes_gt_a[::-1])[::-1]

def print_hit_miss_counters():  
    print("Hit Counters:")
    print("*****")
    print("Hits and Misses by Age:")
    for age in range(len(lifetimes_a)):  
        print(f"Age {age}: Hits - {hits_a[age]}, Evictions - {evictions_a[age]}, \
Lifetimes - {lifetimes_a[age]}, Expected Lifetimes - {expected_lifetimes_a[age]}")
        
    print("Hits > a", hits_gt_a)
    print("Evictions > a", evictions_gt_a) 
    print("Lifetime > a", lifetimes_gt_a)
    
    print("Total Hits = ", sum(hits_a))
    print("Total Misses = ", sum(miss_a))
    
    print("A Hits = ", A_hits)
    print("B Hits = ", B_hits)
    print("\n")
    
# Compute EVA
#def compute_eva():
#    max_age = max(lifetime_a, default=0)
#    expected_lifetimes = upscan(lifetimes_gt_a)
    

# Simulate cache accesses using the alternating access pattern

for _ in range(NUM_ACCESS):
    access_addr = alternate_access_pattern()
    hit = cache_access(access_addr)
        
    if VERBOSE > 2:
        print_cache_contents()
    print("*****")
    print("Eviction Priority:", fifo_queue, "\n")

# Print the current cache contents using the new function
print_cache_contents()
update_statistics()
print_hit_miss_counters()