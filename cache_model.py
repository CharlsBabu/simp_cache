from collections import deque

class CacheLine:
    def __init__(self, addr):
        self.addr = addr
        self.age = 0

class HexDeque(deque):
    def __str__(self):
        return "[" + ", ".join(hex(item) for item in self) + "]"
    
#VERBOSE > 0 : Shows each cache line access with hit and miss info.
VERBOSE = 0

CACHE_SIZE = 4
cache = []
fifo_queue = HexDeque()

# Initialize hit and miss counters for each age
hit_counters = {}
miss_counters = {}

# Arrays A and B information
array_a = {"start_addr": 0x100, "length": 2}
array_b = {"start_addr": 0x200, "length": 6}
current_array = None  # Keep track of the currently accessed array
array_index_a = -1  # Index to keep track of the current position in array A
array_index_b = -1  # Index to keep track of the current position in array B
NUM_ACCESS = 20  # Total number of accesses

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
    if VERBOSE>0:
        print(f"Accessing:  0x{addr:04X}")
    index = find_cache_line(addr)
    # Increment the ages of all cache lines
    increment_cache_ages()

    if index is not None:
        # Cache hit
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
            cache[-1].age = 0  # Set the age of the newly added cache line to 0
            #NOT UPDATING MISS COUNTER IN COMPULSORY MISS
            #update_miss_counters(cache_line.age)
            if VERBOSE>0:
                print(f"Cache miss: Address 0x{addr:04X} added to cache")

    fifo_queue.append(addr)

def find_cache_line(addr):
    for i, cache_line in enumerate(cache):
        if cache_line.addr == addr:
            return i
    return None

def replace_cacheline(index, new_addr):
    old_age = cache[index].age
    old_addr = cache[index].addr
    update_miss_counters(old_age)

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

def update_miss_counters(age):
    if age not in miss_counters:
        miss_counters[age] = 1
    else:
        miss_counters[age] += 1

def print_cache_contents():
    print("Current cache contents:")
    print("*****")
    for i, cache_line in enumerate(cache):
        print(f"Cache Line {i}: Address 0x{cache_line.addr:04X}, Age {cache_line.age}")

# Print hit and miss counters for each age
def print_hit_miss_counters():
    print("Hit Counters:")
    print("*****")
    max_age = max(max(hit_counters.keys(), default=0), max(miss_counters.keys(), default=0))
    print("Hits and Misses by Age:")
    for age in range(max_age + 1):
        hit_count = hit_counters.get(age, 0)
        miss_count = miss_counters.get(age, 0)
        print(f"Age {age}: Hits - {hit_count}, Misses - {miss_count}")
    print("\n")

# Simulate cache accesses using the alternating access pattern
for _ in range(NUM_ACCESS):
    access_address = alternate_access_pattern()
    cache_access(access_address)
    print_cache_contents()
    print("*****")
    print("Eviction Priority:", fifo_queue, "\n")

# Print the current cache contents using the new function
print_cache_contents()
print_hit_miss_counters()
