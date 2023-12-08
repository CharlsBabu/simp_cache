from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random

class CacheLine:
    def __init__(self, addr):
        self.addr = addr
        self.age = 0
        self.eva = 0

class HexDeque(deque):
    def __str__(self):
        return "[" + ", ".join(hex(item) for item in self) + "]"

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
            if REPL=="LRU":
                index_to_replace = find_cache_line(fifo_queue.popleft())
            elif REPL=="EVA":
                index_to_replace = find_min_eva_line()
            elif REPL=="RND":
                index_to_replace = random.randint(0, CACHE_SIZE-1)
            else:    
                raise ValueError("Invalid Repl Policy")
                
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

def find_min_eva_line():
    #update_statistics()
    min_entry = min(cache, key=lambda entry: entry.eva)
    min_index = cache.index(min_entry)
    return min_index
    
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
    global expected_lifetimes_a, EVA, max_age, cost, reward
    
    max_age = max(max(hit_counters.keys(), default=0), 
                  max(miss_counters.keys(), default=0), 
                  max(eviction_counters.keys(), default=0),
                  max(cache, key=lambda entry: entry.age).age)
    
    #cache might not always have the line with the max age, might have got replaced.
    #max_age = max(max_age, max(cache, key=lambda entry: entry.age).age)
    
    hits_a = miss_a = evictions_a = np.array([])
    
    for age in range(max_age+1,-1,-1):
        hits_a = np.insert(hits_a, 0, hit_counters.get(age, 0))
        miss_a = np.insert(miss_a, 0, miss_counters.get(age, 0))
        evictions_a = np.insert(evictions_a, 0, eviction_counters.get(age, 0))
    
    lifetimes_a = hits_a + evictions_a
    
    tot_hits = sum(hits_a)
    N = CACHE_SIZE #sum((np.arange(len(lifetimes_a))+1)*lifetimes_a)/sum(lifetimes_a)
    
    perAccessCost = 0 if sum(lifetimes_a)==0 else tot_hits/(N*sum(lifetimes_a))
    
    EVA = np.array([0.0]*(max_age+2))
    reward = np.array([0.0]*(max_age+2))
    cost = np.array([0.0]*(max_age+2))
    expLifetime = 0.0
    hits2 = 0.0
    events2 = 0.0
    
    for a in range(max_age+1, -1, -1):
        expLifetime += events2
        reward[a] = 0 if events2==0 else hits2 / events2
        cost[a] = 0 if events2==0 else (perAccessCost*expLifetime) / events2
        EVA[a] = reward[a] - cost[a] # if events2==0 else (hits2 - (perAccessCost*expLifetime)) / events2
        
        hits2 += hits_a[a]
        events2 += hits_a[a] + evictions_a[a]
          
    hits_gt_a = upscan(hits_a)
    evictions_gt_a = upscan(evictions_a)
    lifetimes_gt_a = upscan(lifetimes_a) #hits_gt_a + evictions_gt_a
    expected_lifetimes_a = np.cumsum(lifetimes_gt_a[::-1])[::-1]
    
    #events = np.where(lifetimes_gt_a == 0, np.inf, lifetimes_gt_a) #as per the Algorithm 1 in the paper.
    #EVA = (hits_gt_a - (perAccessCost * expected_lifetimes_a)) / events
    
    #assign EVA to respective cachelines based on age.
    for cacheline in cache:
        cacheline.eva = EVA[cacheline.age]#todo
        
def print_hit_miss_counters():  
    print("Hit Counters:")
    print("*****")
    print("Hits and Misses by Age:")
    
    if VERBOSE > 1:
        for age in range(len(lifetimes_a)):  
            print(f"Age {age}: Hits - {hits_a[age]}, Evictions - {evictions_a[age]}, \
Lifetimes - {lifetimes_a[age]}, Expected Lifetimes - {expected_lifetimes_a[age]}")
        
    #print("Hits > a", hits_gt_a)
    #print("Evictions > a", evictions_gt_a) 
    #print("Lifetime > a", lifetimes_gt_a)
    
    total_hits = sum(hits_a)
    print("Total Hits = ", total_hits)
    print("Total Misses = ", sum(miss_a))
    
    hit_rate = total_hits*100/NUM_ACCESS
    print("Hit Rate = ", hit_rate, "%")
    print("A Hits = ", A_hits)
    print("B Hits = ", B_hits)
    
    print("Calculated N", sum((np.arange(len(lifetimes_a))+1)*lifetimes_a)/sum(lifetimes_a))
    #print("EVA = ", EVA)
    print("\n")
    return hit_rate

#VERBOSE > 0 : Shows each cache line access with hit and miss info.
VERBOSE = 0
PLOT = 0

def run_sim(cache_size, asize, bsize, repl_policy):
    global REPL, cache, fifo_queue, hit_counters, miss_counters, eviction_counters, max_age
    global lifetimes_a, hits_a, miss_a, evictions_a, hits_gt_a, evictions_gt_a, lifetimes_gt_a
    global expected_lifetimes_a, EVA, reward, cost, array_a, array_b, a_arr, b_arr, A_hits, B_hits
    global CACHE_SIZE, NUM_ACCESS, array_index_a, array_index_b, current_array
    
    CACHE_SIZE = cache_size
    REPL= repl_policy #"LRU" if 0 else "EVA" 
    cache = []
    fifo_queue = HexDeque()

    # Initialize hit and miss counters for each age 
    hit_counters = {}
    miss_counters = {}
    eviction_counters = {}
    max_age = 0

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
    reward = np.array([])
    cost = np.array([])

    # Arrays A and B information
    array_a = {"start_addr": 0xA000, "length": asize}
    array_b = {"start_addr": 0xB000, "length": bsize}
    a_arr = np.arange(array_a["start_addr"], array_a["start_addr"]+array_a["length"])
    b_arr = np.arange(array_b["start_addr"], array_b["start_addr"]+array_b["length"])
    A_hits = 0
    B_hits = 0

    current_array = None  # Keep track of the currently accessed array
    array_index_a = -1  # Index to keep track of the current position in array A
    array_index_b = -1  # Index to keep track of the current position in array B
    NUM_ACCESS = 5000  # Total number of accesses
    EVA_UPDATE_INTERVAL = 5

# Simulate cache accesses using the alternating access pattern

    for i in range(NUM_ACCESS):
        access_addr = alternate_access_pattern()
        #print(i)
        cache_access(access_addr)
            
        if VERBOSE > 1:
            print_cache_contents()
            print("*****")
        if i%EVA_UPDATE_INTERVAL==0 and i>0:
            update_statistics()
        
        if VERBOSE > 2:
            if REPL=="LRU":
                print("Eviction Priority:", fifo_queue, "\n")
            elif REPL=="EVA":
                print("EVA:", EVA, "\n")

    update_statistics()
    hit_rate = print_hit_miss_counters()
    
    if PLOT:
        plt.plot(EVA, label="EVA")
        plt.plot(reward, label="reward")
        plt.plot(cost, label="cost")
        plt.legend(loc="lower right")
        plt.show()
    
    return hit_rate

def ideal_HR(cache_size, asize, bsize):
    compulsory_misses = cache_size
    a_hitrate = min(cache_size, asize)/asize
    b_hitrate = max(0, cache_size-asize)/bsize
    
    A_hits = int(a_hitrate*(0.5*NUM_ACCESS))
    B_hits = int(b_hitrate*(0.5*NUM_ACCESS))
    total_hits = A_hits + B_hits - compulsory_misses
    
    print("Total Hits = ", total_hits)
    print("Total Misses = ", NUM_ACCESS-total_hits)
    
    hit_rate = total_hits*100/NUM_ACCESS
    print("Hit Rate = ", hit_rate, "%")
    print("A Hits = ", A_hits)
    print("B Hits = ", B_hits)
    print("\n")
    
    return hit_rate
    
###########################################################
#SIMULATION

CACHE_SIZE = 64
FIXED_SZ = 1024 #128, 256, 512, 1024
sweep = np.arange(8, 128, 4)

HR_lru = np.array([0.0]*len(sweep))
HR_eva = np.array([0.0]*len(sweep))
HR_rnd = np.array([0.0]*len(sweep))
HR_ideal = np.array([0.0]*len(sweep))

for i,size in enumerate(sweep):
    print("Run ", i, "size = ", size)
    HR_lru[i] = run_sim(cache_size=CACHE_SIZE, asize=size, bsize=FIXED_SZ, repl_policy="LRU")
    HR_eva[i] = run_sim(cache_size=CACHE_SIZE, asize=size, bsize=FIXED_SZ, repl_policy="EVA")
    HR_rnd[i] = run_sim(cache_size=CACHE_SIZE, asize=size, bsize=FIXED_SZ, repl_policy="RND")
    HR_ideal[i] = ideal_HR(cache_size=CACHE_SIZE, asize=size, bsize=FIXED_SZ)
    
plt.plot(sweep, HR_lru, label="LRU")
plt.plot(sweep, HR_eva, label="EVA")
for i, j in zip(sweep[2::4], HR_eva[2::4]):
     plt.annotate('(%s, %s)' % (i, j), xy=(i, j), textcoords='offset points', xytext=(0,10), ha='center')

plt.plot(sweep, HR_rnd, label="RND")
plt.plot(sweep, HR_ideal, label="ideal")

plt.legend(loc="upper right")
plt.title("Hit Rate vs Small Array Size, CACHE= %d "% CACHE_SIZE + "FIXED_ARR=%d" %FIXED_SZ)
plt.grid()
plt.show()



