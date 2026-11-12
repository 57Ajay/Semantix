# Interview feedback: backend engineer, August 2024

**Recommendation:** hire, strong yes on system design.

## System design (60 min)

Asked to design a URL shortener that handles 10,000 writes per second. Started with
requirements and rough numbers before drawing anything, which was good. Chose a
counter-based id with base62 encoding, explained why random ids need collision checks.
Put a cache in front of reads and discussed what happens when the cache is cold.

Weak spot: did not think about abuse (spam links) until prompted.

## Coding (45 min)

Implemented an LRU cache with a dictionary and a doubly linked list. Clean code, wrote
tests without being asked. Took a while to handle the capacity-zero edge case.

## Culture

Asked good questions about on-call load and how we write postmortems.
