## Leg joint velocities vs the PhysX velocity limits read back (repeat 1)

Max |joint velocity| over all phases of each trial, and the number of command-phase steps at >= 99 % of the PhysX limit (the limits are the URDF values; the configs' velocity_limit is not written by the fork).

| Robot | Joint | PhysX limit (rad/s) | stand: max / steps at limit | forward: max / steps at limit | turn_left: max / steps at limit | turn_right: max / steps at limit | sequence: max / steps at limit |
|---|---|---|---|---|---|---|---|
| E | Left_Hip_Pitch | 7.10 | 2.31 (reset warmup) / 0 of 600 | 7.10 (command) / 17 of 600 | 5.08 (command) / 0 of 600 | 6.12 (command) / 0 of 600 | 7.10 (command) / 11 of 1500 |
| E | Right_Hip_Pitch | 7.10 | 3.75 (reset warmup) / 0 of 600 | 3.75 (reset warmup) / 0 of 600 | 3.75 (reset warmup) / 0 of 600 | 3.75 (reset warmup) / 0 of 600 | 5.40 (command) / 0 of 1500 |
| E | Left_Hip_Roll | 12.90 | 1.58 (reset warmup) / 0 of 600 | 2.54 (command) / 0 of 600 | 2.03 (command) / 0 of 600 | 2.25 (command) / 0 of 600 | 3.68 (command) / 0 of 1500 |
| E | Right_Hip_Roll | 12.90 | 1.58 (reset warmup) / 0 of 600 | 3.35 (command) / 0 of 600 | 2.73 (command) / 0 of 600 | 2.43 (command) / 0 of 600 | 4.31 (command) / 0 of 1500 |
| E | Left_Hip_Yaw | 18.10 | 2.42 (reset warmup) / 0 of 600 | 3.00 (command) / 0 of 600 | 2.77 (command) / 0 of 600 | 6.75 (command) / 0 of 600 | 3.90 (command) / 0 of 1500 |
| E | Right_Hip_Yaw | 18.10 | 1.63 (reset warmup) / 0 of 600 | 3.73 (command) / 0 of 600 | 3.64 (command) / 0 of 600 | 3.72 (command) / 0 of 600 | 4.08 (command) / 0 of 1500 |
| E | Left_Knee_Pitch | 12.50 | 6.33 (reset warmup) / 0 of 600 | 11.27 (command) / 0 of 600 | 9.69 (command) / 0 of 600 | 9.38 (command) / 0 of 600 | 11.36 (command) / 0 of 1500 |
| E | Right_Knee_Pitch | 12.50 | 5.91 (reset warmup) / 0 of 600 | 6.26 (command) / 0 of 600 | 5.91 (reset warmup) / 0 of 600 | 5.91 (reset warmup) / 0 of 600 | 8.92 (command) / 0 of 1500 |
| E | Left_Ankle_Pitch | 18.10 | 14.03 (reset warmup) / 0 of 600 | 14.03 (reset warmup) / 0 of 600 | 14.03 (reset warmup) / 0 of 600 | 14.03 (reset warmup) / 0 of 600 | 14.03 (reset warmup) / 0 of 1500 |
| E | Right_Ankle_Pitch | 18.10 | 3.92 (reset warmup) / 0 of 600 | 12.15 (command) / 0 of 600 | 11.72 (command) / 0 of 600 | 11.57 (command) / 0 of 600 | 12.96 (command) / 0 of 1500 |
| E | Left_Ankle_Roll | 18.10 | 4.78 (reset warmup) / 0 of 600 | 6.13 (command) / 0 of 600 | 4.78 (reset warmup) / 0 of 600 | 5.43 (command) / 0 of 600 | 8.20 (command) / 0 of 1500 |
| E | Right_Ankle_Roll | 18.10 | 3.35 (reset warmup) / 0 of 600 | 3.35 (reset warmup) / 0 of 600 | 3.35 (reset warmup) / 0 of 600 | 3.35 (reset warmup) / 0 of 600 | 6.19 (command) / 0 of 1500 |
| T | Left_Hip_Pitch | 7.10 | 2.37 (reset warmup) / 0 of 600 | 7.09 (command) / 17 of 600 | 4.98 (command) / 0 of 600 | 5.99 (command) / 0 of 600 | 7.09 (command) / 10 of 1500 |
| T | Right_Hip_Pitch | 7.10 | 2.97 (reset warmup) / 0 of 600 | 2.97 (reset warmup) / 0 of 600 | 3.48 (command) / 0 of 600 | 3.24 (command) / 0 of 600 | 5.48 (command) / 0 of 1500 |
| T | Left_Hip_Roll | 12.90 | 1.85 (reset warmup) / 0 of 600 | 2.33 (command) / 0 of 600 | 2.04 (command) / 0 of 600 | 2.21 (command) / 0 of 600 | 2.93 (command) / 0 of 1500 |
| T | Right_Hip_Roll | 12.90 | 1.61 (reset warmup) / 0 of 600 | 3.12 (command) / 0 of 600 | 2.80 (command) / 0 of 600 | 2.29 (command) / 0 of 600 | 4.03 (command) / 0 of 1500 |
| T | Left_Hip_Yaw | 18.10 | 2.42 (reset warmup) / 0 of 600 | 3.12 (command) / 0 of 600 | 3.04 (command) / 0 of 600 | 5.07 (command) / 0 of 600 | 4.05 (command) / 0 of 1500 |
| T | Right_Hip_Yaw | 18.10 | 1.62 (reset warmup) / 0 of 600 | 3.75 (command) / 0 of 600 | 3.70 (command) / 0 of 600 | 3.12 (command) / 0 of 600 | 4.25 (command) / 0 of 1500 |
| T | Left_Knee_Pitch | 12.50 | 6.34 (reset warmup) / 0 of 600 | 11.43 (command) / 0 of 600 | 10.00 (command) / 0 of 600 | 9.40 (command) / 0 of 600 | 11.43 (command) / 0 of 1500 |
| T | Right_Knee_Pitch | 12.50 | 5.36 (reset warmup) / 0 of 600 | 6.26 (command) / 0 of 600 | 5.36 (reset warmup) / 0 of 600 | 5.36 (reset warmup) / 0 of 600 | 8.90 (command) / 0 of 1500 |
| T | Left_Ankle_Pitch | 18.10 | 12.64 (reset warmup) / 0 of 600 | 12.64 (reset warmup) / 0 of 600 | 12.64 (reset warmup) / 0 of 600 | 12.64 (reset warmup) / 0 of 600 | 12.64 (reset warmup) / 0 of 1500 |
| T | Right_Ankle_Pitch | 18.10 | 8.29 (reset warmup) / 0 of 600 | 12.12 (command) / 0 of 600 | 11.92 (command) / 0 of 600 | 11.36 (command) / 0 of 600 | 12.77 (command) / 0 of 1500 |
| T | Left_Ankle_Roll | 18.10 | 4.78 (reset warmup) / 0 of 600 | 6.10 (command) / 0 of 600 | 4.78 (reset warmup) / 0 of 600 | 5.16 (command) / 0 of 600 | 6.36 (command) / 0 of 1500 |
| T | Right_Ankle_Roll | 18.10 | 1.87 (reset warmup) / 0 of 600 | 2.76 (command) / 0 of 600 | 2.65 (command) / 0 of 600 | 2.26 (command) / 0 of 600 | 6.02 (command) / 0 of 1500 |
