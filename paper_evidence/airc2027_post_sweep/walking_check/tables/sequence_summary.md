## Executor-like sequence: achieved vs implied per complete segment

| Segment type | Quantity | Implied | E: n, mean ± SD [min, max] | E ratio | T: n, mean ± SD [min, max] | T ratio |
|---|---|---|---|---|---|---|
| forward | displacement along start heading (m) | 0.75 | 9, 0.610 ± 0.013 [0.590, 0.625] | 0.813 | 9, 0.610 ± 0.011 [0.599, 0.629] | 0.814 |
| forward | same, at end of following zero segment (m) | 0.75 | 9, 0.651 ± 0.013 [0.630, 0.667] | 0.868 | 9, 0.651 ± 0.010 [0.641, 0.669] | 0.869 |
| forward | lateral displacement (m) | 0 | 9, 0.001 ± 0.036 [-0.043, 0.080] | n/a | 9, -0.010 ± 0.036 [-0.048, 0.074] | n/a |
| forward | yaw change (deg) | 0 | 9, 1.555 ± 1.103 [0.631, 4.307] | n/a | 9, 0.542 ± 1.976 [-3.408, 3.976] | n/a |
| turn_left | yaw change (deg) | 45 | 5, 42.987 ± 0.854 [41.952, 44.036] | 0.955 | 5, 43.952 ± 0.958 [42.867, 44.906] | 0.977 |
| turn_left | same, at end of following zero segment (deg) | 45 | 5, 44.851 ± 0.890 [43.542, 45.737] | 0.997 | 5, 46.087 ± 1.234 [44.599, 47.561] | 1.024 |
| turn_left | planar displacement (m) | 0 | 5, 0.039 ± 0.035 [0.007, 0.091] | n/a | 5, 0.037 ± 0.035 [0.013, 0.095] | n/a |
| turn_right | yaw change (deg) | -45 | 4, -42.966 ± 0.531 [-43.467, -42.343] | 0.955 | 4, -42.960 ± 1.627 [-44.523, -41.165] | 0.955 |
| turn_right | same, at end of following zero segment (deg) | -45 | 4, -43.318 ± 0.623 [-44.060, -42.562] | 0.963 | 4, -42.402 ± 1.427 [-43.704, -41.059] | 0.942 |
| turn_right | planar displacement (m) | 0 | 4, 0.028 ± 0.017 [0.007, 0.046] | n/a | 4, 0.039 ± 0.008 [0.032, 0.049] | n/a |
| zero after forward (0.16 s) | residual planar displacement (m) | 0 | 9, 0.041 ± 0.001 [0.039, 0.043] |  | 9, 0.041 ± 0.002 [0.038, 0.044] |  |
| zero after forward (0.16 s) | residual absolute yaw change (deg) | 0 | 9, 0.361 ± 0.239 [0.088, 0.863] |  | 9, 0.315 ± 0.306 [0.003, 0.970] |  |
| zero after forward (0.16 s) | mean forward speed (m/s) | 0 | 9, 0.240 ± 0.012 [0.213, 0.251] |  | 9, 0.240 ± 0.009 [0.228, 0.256] |  |
| zero after forward (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 9, 0.135 ± 0.034 [0.096, 0.205] |  | 9, 0.178 ± 0.037 [0.109, 0.230] |  |
| zero after turn left (0.16 s) | residual planar displacement (m) | 0 | 5, 0.014 ± 0.001 [0.013, 0.016] |  | 5, 0.014 ± 0.002 [0.012, 0.016] |  |
| zero after turn left (0.16 s) | residual absolute yaw change (deg) | 0 | 5, 2.062 ± 1.047 [0.495, 3.297] |  | 5, 2.135 ± 1.045 [0.894, 3.637] |  |
| zero after turn left (0.16 s) | mean forward speed (m/s) | 0 | 5, -0.088 ± 0.025 [-0.115, -0.049] |  | 5, -0.094 ± 0.025 [-0.126, -0.063] |  |
| zero after turn left (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 5, 0.233 ± 0.058 [0.183, 0.317] |  | 5, 0.256 ± 0.085 [0.172, 0.352] |  |
| zero after turn right (0.16 s) | residual planar displacement (m) | 0 | 4, 0.015 ± 0.003 [0.012, 0.019] |  | 4, 0.014 ± 0.005 [0.007, 0.018] |  |
| zero after turn right (0.16 s) | residual absolute yaw change (deg) | 0 | 4, 0.508 ± 0.809 [0.003, 1.717] |  | 4, 0.615 ± 0.370 [0.114, 0.955] |  |
| zero after turn right (0.16 s) | mean forward speed (m/s) | 0 | 4, -0.118 ± 0.016 [-0.133, -0.099] |  | 4, -0.106 ± 0.029 [-0.126, -0.063] |  |
| zero after turn right (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 4, 0.177 ± 0.042 [0.135, 0.226] |  | 4, 0.224 ± 0.018 [0.212, 0.250] |  |

RMS error against the step-wise command over the 30 s — E: vx 0.140 m/s, vy 0.085 m/s, wz 0.142 rad/s; T: vx 0.134 m/s, vy 0.078 m/s, wz 0.178 rad/s.

