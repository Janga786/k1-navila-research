# K1_locomotion.urdf (evaluated) vs K1_22dof.urdf (training)

SHA-256: K1_locomotion.urdf `5cf5951d…`, K1_22dof.urdf `9312ecc9…` (full hashes in `../provenance_sweep_files.txt`).
`diff` of the two files: the only differences are the 10 head/arm joint `type` attributes (revolute → fixed) and two
XML comment delimiters. Origins, axes, limits, masses, inertias, visuals and collisions are byte-identical.

## Head and arm joints

| Joint | Parent → child | K1_locomotion | K1_22dof | origin xyz | origin rpy | axis | training pose (rad) |
|---|---|---|---|---|---|---|---|
| AAHead_yaw | Trunk → Head_1 | fixed | revolute | [0.0056, 0.0, 0.2149] | [0.0, 0.0, 0.0] | [0.0, 0.0, 1.0] | 0.0 |
| Head_pitch | Head_1 → Head_2 | fixed | revolute | [0.0, 0.0, 0.033] | [0.0, 0.0, 0.0] | [0.0, 1.0, 0.0] | 0.0 |
| ALeft_Shoulder_Pitch | Trunk → Left_Arm_1 | fixed | revolute | [0.0, 0.077, 0.1845] | [0.0, 0.0, 0.0] | [0.0, 1.0, 0.0] | 0.2 |
| Left_Shoulder_Roll | Left_Arm_1 → Left_Arm_2 | fixed | revolute | [0.0025, 0.068, -0.0135] | [0.0, 0.0, 0.0] | [1.0, 0.0, 0.0] | -1.25 |
| Left_Elbow_Pitch | Left_Arm_2 → Left_Arm_3 | fixed | revolute | [0.0, 0.044428, 0.0] | [0.0, 0.0, 0.0] | [0.0, 1.0, 0.0] | 0.0 |
| Left_Elbow_Yaw | Left_Arm_3 → left_hand_link | fixed | revolute | [0.0, 0.1215, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 1.0] | -0.5 |
| ARight_Shoulder_Pitch | Trunk → Right_Arm_1 | fixed | revolute | [0.0, -0.077, 0.1845] | [0.0, 0.0, 0.0] | [0.0, 1.0, 0.0] | 0.2 |
| Right_Shoulder_Roll | Right_Arm_1 → Right_Arm_2 | fixed | revolute | [0.0025, -0.068, -0.0135] | [0.0, 0.0, 0.0] | [1.0, 0.0, 0.0] | 1.25 |
| Right_Elbow_Pitch | Right_Arm_2 → Right_Arm_3 | fixed | revolute | [0.0, -0.044428, 0.0] | [0.0, 0.0, 0.0] | [0.0, 1.0, 0.0] | 0.0 |
| Right_Elbow_Yaw | Right_Arm_3 → right_hand_link | fixed | revolute | [0.0, -0.1215, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 1.0] | 0.5 |

With every joint fixed at its origin (angle 0), the arm pose is the zero pose: shoulder roll 0 puts the
arms straight out sideways (T-pose). Link frame positions in the Trunk frame:

| Link | zero pose (evaluated) xyz | training pose xyz |
|---|---|---|
| Head_1 | [0.0056, 0.0, 0.2149] | [0.0056, 0.0, 0.2149] |
| Head_2 | [0.0056, 0.0, 0.2479] | [0.0056, 0.0, 0.2479] |
| Left_Arm_1 | [0.0, 0.077, 0.1845] | [0.0, 0.077, 0.1845] |
| Left_Arm_2 | [0.0025, 0.145, 0.171] | [-0.0002, 0.145, 0.1708] |
| Left_Arm_3 | [0.0025, 0.1894, 0.171] | [-0.0086, 0.159, 0.1295] |
| left_hand_link | [0.0025, 0.3109, 0.171] | [-0.0315, 0.1973, 0.0164] |
| Right_Arm_1 | [0.0, -0.077, 0.1845] | [0.0, -0.077, 0.1845] |
| Right_Arm_2 | [0.0025, -0.145, 0.171] | [-0.0002, -0.145, 0.1708] |
| Right_Arm_3 | [0.0025, -0.1894, 0.171] | [-0.0086, -0.159, 0.1295] |
| right_hand_link | [0.0025, -0.3109, 0.171] | [-0.0315, -0.1973, 0.0164] |

## Masses

| Link | mass (kg), both files |
|---|---|
| Trunk | 6.5 |
| Head_1 | 0.3 |
| Head_2 | 0.7 |
| Left_Arm_1 | 0.5 |
| Left_Arm_2 | 0.09 |
| Left_Arm_3 | 0.8 |
| left_hand_link | 0.19 |
| Right_Arm_1 | 0.5 |
| Right_Arm_2 | 0.09 |
| Right_Arm_3 | 0.8 |
| right_hand_link | 0.19 |
| Left_Hip_Pitch | 0.69 |
| Left_Hip_Roll | 0.13 |
| Left_Hip_Yaw | 1.65 |
| Left_Shank | 1.5 |
| Left_Ankle_Cross | 0.039 |
| left_foot_link | 0.494 |
| Right_Hip_Pitch | 0.69 |
| Right_Hip_Roll | 0.13 |
| Right_Hip_Yaw | 1.65 |
| Right_Shank | 1.5 |
| Right_Ankle_Cross | 0.039 |
| right_foot_link | 0.494 |
| **total** | **19.666** (K1_22dof: 19.666) |

## Rigid merge (what the evaluated Trunk body should be)

Trunk + Head_1 + Head_2 + both arm chains at the zero pose: mass **10.660 kg**, COM [-0.001, -0.0005, 0.1196] (Trunk frame),
inertia about the COM (Trunk axes) diag ≈ [0.38, 0.1609, 0.2368], principal moments [0.1609, 0.2365, 0.3802].
Trunk link alone: 6.5 kg, diag [0.0962, 0.0895, 0.0202].
