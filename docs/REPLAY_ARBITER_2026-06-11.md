# fp16 replay arbiter — results

| ep | ticks | int8-replay==orig | fp16==orig | fp16==int8-replay | fp16 action==int8 action |
|---|---|---|---|---|---|
| ep0 | 77 | 58/77 | 61/77 | 70/77 | 70/77 |
| ep1 | 57 | 50/57 | 45/57 | 48/57 | 48/57 |
| ep2 | 61 | 49/61 | 46/61 | 54/61 | 54/61 |
| ep3 | 16 | 15/16 | 14/16 | 15/16 | 15/16 |
| ep4 | 24 | 22/24 | 23/24 | 23/24 | 23/24 |
| ep5 | 39 | 38/39 | 38/39 | 39/39 | 39/39 |
| ep6 | 43 | 37/43 | 39/43 | 38/43 | 38/43 |
| ep7 | 25 | 24/25 | 24/25 | 25/25 | 25/25 |
| ep8 | 16 | 14/16 | 14/16 | 16/16 | 16/16 |
| ep9 | 66 | 54/66 | 54/66 | 66/66 | 66/66 |

**TOTALS over 424 ticks:**
- int8-replay == int8-original (determinism + jpg-vs-png): 361/424 = 85.1%
- fp16-replay == int8-original (verbatim): 358/424 = 84.4%
- fp16-replay == int8-replay (verbatim, identical inputs): 394/424 = 92.9%
- fp16 vs int8 ACTION-level match (what the robot would do): 394/424 = 92.9%
- fp16 vs ORIGINAL action-level match: 358/424 = 84.4%

## Action-level disagreements (fp16 vs original), 65 total
- ep0 tick 11: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is move forward 25 cm.'
- ep0 tick 20: int8 'The next action is turn right 30 degree.'  ->  fp16 'The next action is turn right 45 degree.'
- ep0 tick 21: int8 'The next action is turn left 15 degree.'  ->  fp16 'The next action is move forward 25 cm.'
- ep0 tick 23: int8 'The next action is turn right 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ep0 tick 24: int8 'The next action is turn right 30 degree.'  ->  fp16 'The next action is turn right 45 degree.'
- ep0 tick 27: int8 'The next action is move forward 25 cm.'  ->  fp16 'The next action is turn left 15 degree.'
- ep0 tick 35: int8 'The next action is turn left 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 36: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 39: int8 'The next action is move forward 25 cm.'  ->  fp16 'The next action is move forward 75 cm.'
- ep0 tick 43: int8 'The next action is turn left 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 51: int8 'The next action is move forward 25 cm.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 57: int8 'The next action is turn left 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 58: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ep0 tick 70: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is move forward 50 cm.'
- ep0 tick 73: int8 'The next action is turn left 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep0 tick 74: int8 'The next action is move forward 25 cm.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 0: int8 'The next action is turn left 30 degree.'  ->  fp16 'The next action is move forward 25 cm.'
- ep1 tick 6: int8 'The next action is turn right 30 degree.'  ->  fp16 'The next action is turn left 30 degree.'
- ep1 tick 11: int8 'The next action is move forward 25 cm.'  ->  fp16 'The next action is turn left 15 degree.'
- ep1 tick 13: int8 'The next action is turn right 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ep1 tick 14: int8 'The next action is turn right 45 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 16: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is move forward 25 cm.'
- ep1 tick 35: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 36: int8 'The next action is turn right 45 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 37: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 40: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep1 tick 41: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is move forward 25 cm.'
- ep1 tick 49: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is turn left 45 degree.'
- ep2 tick 4: int8 'The next action is turn right 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ep2 tick 11: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is move forward 75 cm.'
- ep2 tick 14: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is move forward 25 cm.'
- ep2 tick 15: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn left 30 degree.'
- ep2 tick 17: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is turn right 15 degree.'
- ep2 tick 21: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn right 45 degree.'
- ep2 tick 24: int8 'The next action is move forward 75 cm.'  ->  fp16 'The next action is move forward 25 cm.'
- ep2 tick 28: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn right 45 degree.'
- ep2 tick 32: int8 'The next action is turn right 15 degree.'  ->  fp16 'The next action is turn right 45 degree.'
- ep2 tick 34: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn left 30 degree.'
- ep2 tick 39: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ep2 tick 44: int8 'The next action is turn left 45 degree.'  ->  fp16 'The next action is turn left 15 degree.'
- ... and 25 more
