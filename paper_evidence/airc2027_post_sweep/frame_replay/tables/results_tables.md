## Episodes

| episode_idx | record | run_A | run_B | run_C | run_D | pair | qstar_index | qstar_step_A | first_state_diff | first_request_diff_index | frames_differ_at_qstar | qstar_max_abs_diff | reply_A | reply_X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 6 | 1022 steps, stop, success=0.0 | 903 steps, stop, success=0.0 |  |  | A vs B | 3 | 109 | none before q* | 0 | 8 of 8 | 11 | The next action is move forward 25 cm. | The next action is turn right 15 degree. |
| 8 | 11 | 1573 steps, sim_done, success=0.0 | 1776 steps, stop, success=1.0 |  |  | A vs B | 6 | 530 | none before q* | 0 | 8 of 8 | 19 | The next action is turn right 15 degree. | The next action is move forward 25 cm. |
| 16 | 25 | 6001 steps, step_cap, success=0.0 | 2232 steps, sim_done, success=0.0 |  |  | A vs B | 20 | 1807 | none before q* | 0 | 8 of 8 | 15 | The next action is turn right 45 degree. | The next action is move forward 25 cm. |
| 20 | 32 | 1563 steps, sim_done, success=0.0 | 1437 steps, stop, success=0.0 |  |  | A vs B | 4 | 277 | none before q* | 0 | 8 of 8 | 12 | The next action is turn left 30 degree. | The next action is turn left 45 degree. |
| 50 | 77 | 6001 steps, step_cap, success=0.0 | 6001 steps, step_cap, success=0.0 |  |  | A vs B | 51 | 5482 | none before q* | 0 | 8 of 8 | 65 | The next action is turn right 15 degree. | The next action is turn left 45 degree. |
| 99 | 144 | 2677 steps, stop, success=0.0 | 1947 steps, stop, success=1.0 |  |  | A vs B | 11 | 724 | none before q* | 0 | 8 of 8 | 39 | The next action is turn right 15 degree. | The next action is move forward 25 cm. |
| 92 | 137 | 1025 steps, stop, success=1.0 | 926 steps, stop, success=1.0 |  |  | A vs B | 10 | 665 | none before q* | 0 | 8 of 8 | 31 | The next action is turn right 15 degree. | The next action is turn right 30 degree. |
| 212 | 323 | 1049 steps, stop, success=1.0 | 994 steps, stop, success=1.0 |  |  | A vs B | 13 | 954 | none before q* | 0 | 8 of 8 | 24 | The next action is move forward 75 cm. | The next action is move forward 25 cm. |

## Hypotheses

| episode_idx | H1 | H1_detail | H2_cond1 | H2_cond2 | H2_cond3 | H3 | H3_first_mismatch |
|---|---|---|---|---|---|---|---|
| 3 | held | a=True b=True c=True; first state diff 109 | held | held | held | held |  |
| 8 | held | a=True b=True c=True; first state diff 530 | held | held | held | held |  |
| 16 | held | a=True b=True c=True; first state diff 1807 | held | held | held | held |  |
| 20 | held | a=True b=True c=True; first state diff 321 | held | held | held | held |  |
| 50 | held | a=True b=True c=True; first state diff 5482 | held | held | held | held |  |
| 99 | held | a=True b=True c=True; first state diff 724 | held | held | held | held |  |
| 92 | held | a=True b=True c=True; first state diff 692 | held | held | held | held |  |
| 212 | held | a=True b=True c=True; first state diff 986 | held | held | held | held |  |

Totals: {"H1": "8 of 8", "H3": "8 of 8", "H2_cond1": "8 of 8", "H2_cond2": "8 of 8", "H2_cond3": "8 of 8"}

## H2 reply counts

| episode_idx | condition | request | own_reply_count | sends | distribution |
|---|---|---|---|---|---|
| 3 | cond1 | ep3_A_q003 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 3 | cond1 | ep3_B_q003 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 3 | cond2 | ep3_A_q003 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 3 | cond2 | ep3_B_q003 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 3 | cond3 | ep3_A_q003 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 3 | cond3 | ep3_B_q003 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 8 | cond1 | ep8_A_q006 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 8 | cond1 | ep8_B_q006 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 8 | cond2 | ep8_A_q006 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 8 | cond2 | ep8_B_q006 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 8 | cond3 | ep8_A_q006 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 8 | cond3 | ep8_B_q006 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 16 | cond1 | ep16_A_q020 | 20 | 20 | {'The next action is turn right 45 degree.': 20} |
| 16 | cond1 | ep16_B_q020 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 16 | cond2 | ep16_A_q020 | 20 | 20 | {'The next action is turn right 45 degree.': 20} |
| 16 | cond2 | ep16_B_q020 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 16 | cond3 | ep16_A_q020 | 20 | 20 | {'The next action is turn right 45 degree.': 20} |
| 16 | cond3 | ep16_B_q020 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 20 | cond1 | ep20_A_q004 | 20 | 20 | {'The next action is turn left 30 degree.': 20} |
| 20 | cond1 | ep20_B_q004 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 20 | cond2 | ep20_A_q004 | 20 | 20 | {'The next action is turn left 30 degree.': 20} |
| 20 | cond2 | ep20_B_q004 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 20 | cond3 | ep20_A_q004 | 20 | 20 | {'The next action is turn left 30 degree.': 20} |
| 20 | cond3 | ep20_B_q004 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 50 | cond1 | ep50_A_q051 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 50 | cond1 | ep50_B_q051 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 50 | cond2 | ep50_A_q051 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 50 | cond2 | ep50_B_q051 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 50 | cond3 | ep50_A_q051 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 50 | cond3 | ep50_B_q051 | 20 | 20 | {'The next action is turn left 45 degree.': 20} |
| 99 | cond1 | ep99_A_q011 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 99 | cond1 | ep99_B_q011 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 99 | cond2 | ep99_A_q011 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 99 | cond2 | ep99_B_q011 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 99 | cond3 | ep99_A_q011 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 99 | cond3 | ep99_B_q011 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 92 | cond1 | ep92_A_q010 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 92 | cond1 | ep92_B_q010 | 20 | 20 | {'The next action is turn right 30 degree.': 20} |
| 92 | cond2 | ep92_A_q010 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 92 | cond2 | ep92_B_q010 | 20 | 20 | {'The next action is turn right 30 degree.': 20} |
| 92 | cond3 | ep92_A_q010 | 20 | 20 | {'The next action is turn right 15 degree.': 20} |
| 92 | cond3 | ep92_B_q010 | 20 | 20 | {'The next action is turn right 30 degree.': 20} |
| 212 | cond1 | ep212_A_q013 | 20 | 20 | {'The next action is move forward 75 cm.': 20} |
| 212 | cond1 | ep212_B_q013 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 212 | cond2 | ep212_A_q013 | 20 | 20 | {'The next action is move forward 75 cm.': 20} |
| 212 | cond2 | ep212_B_q013 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |
| 212 | cond3 | ep212_A_q013 | 20 | 20 | {'The next action is move forward 75 cm.': 20} |
| 212 | cond3 | ep212_B_q013 | 20 | 20 | {'The next action is move forward 25 cm.': 20} |

## Requests vs replies before q*

| episode_idx | pair | queries_before_qstar | requests_differed_replies_matched_before_qstar | first_request_diff_index |
|---|---|---|---|---|
| 3 | A vs B | 3 | 3 | 0 |
| 8 | A vs B | 6 | 6 | 0 |
| 16 | A vs B | 20 | 20 | 0 |
| 20 | A vs B | 4 | 4 | 0 |
| 50 | A vs B | 51 | 51 | 0 |
| 99 | A vs B | 11 | 11 | 0 |
| 92 | A vs B | 10 | 10 | 0 |
| 212 | A vs B | 13 | 13 | 0 |

## Frames at q*

| episode_idx | slot | k_A | label_A | frame_sha_same | max_abs_diff | mean_abs_diff | share_channels_differ |
|---|---|---|---|---|---|---|---|
| 3 | 1 | 0 | init | False | 11 | 0.15828197337962963 | 0.15587022569444445 |
| 3 | 2 | 1 | w24 | False | 9 | 0.15410662615740742 | 0.15276765046296295 |
| 3 | 3 | 3 | w74 | False | 4 | 0.15589445891203704 | 0.15528175636574074 |
| 3 | 4 | 5 | w124 | False | 4 | 0.18558702256944445 | 0.18333297164351853 |
| 3 | 5 | 7 | w174 | False | 4 | 0.20647786458333334 | 0.2004072627314815 |
| 3 | 6 | 9 | 0 | False | 4 | 0.22086263020833333 | 0.21064670138888889 |
| 3 | 7 | 11 | 50 | False | 5 | 0.14360279224537037 | 0.14308666087962962 |
| 3 | 8 | 13 | 100 | False | 8 | 0.15701244212962964 | 0.15528971354166668 |
| 8 | 1 | 0 | init | False | 12 | 0.16952003761574075 | 0.1641623263888889 |
| 8 | 2 | 4 | w99 | False | 3 | 0.15501880787037037 | 0.15412398726851853 |
| 8 | 3 | 8 | w199 | False | 3 | 0.15385416666666665 | 0.15316297743055557 |
| 8 | 4 | 12 | 75 | False | 17 | 0.16205258969907407 | 0.1575929542824074 |
| 8 | 5 | 17 | 200 | False | 11 | 0.16178819444444445 | 0.15582935474537038 |
| 8 | 6 | 21 | 300 | False | 19 | 0.13940176504629628 | 0.13438838252314814 |
| 8 | 7 | 25 | 400 | False | 16 | 0.11639286747685185 | 0.11392578125 |
| 8 | 8 | 30 | 525 | False | 11 | 0.12397135416666667 | 0.12250940393518518 |
| 16 | 1 | 0 | init | False | 8 | 0.08284830729166667 | 0.08245985243055555 |
| 16 | 2 | 11 | 50 | False | 6 | 0.09038519965277778 | 0.0891648582175926 |
| 16 | 3 | 23 | 350 | False | 5 | 0.10741319444444444 | 0.10695421006944444 |
| 16 | 4 | 34 | 625 | False | 7 | 0.10742513020833333 | 0.10679542824074074 |
| 16 | 5 | 46 | 925 | False | 15 | 0.10816080729166666 | 0.10691984953703704 |
| 16 | 6 | 57 | 1200 | False | 6 | 0.09758282696759259 | 0.09695276331018518 |
| 16 | 7 | 69 | 1500 | False | 10 | 0.12520724826388888 | 0.12373734085648148 |
| 16 | 8 | 81 | 1800 | False | 8 | 0.11355794270833333 | 0.11172272858796296 |
| 20 | 1 | 0 | init | False | 10 | 0.16892939814814814 | 0.16803819444444446 |
| 20 | 2 | 2 | w49 | False | 4 | 0.166455078125 | 0.16610785590277777 |
| 20 | 3 | 5 | w124 | False | 3 | 0.17171115451388888 | 0.17122938368055557 |
| 20 | 4 | 8 | w199 | False | 4 | 0.19493019386574073 | 0.19249240451388888 |
| 20 | 5 | 11 | 50 | False | 3 | 0.15081633391203703 | 0.1506611689814815 |
| 20 | 6 | 14 | 125 | False | 3 | 0.143916015625 | 0.14379991319444443 |
| 20 | 7 | 17 | 200 | False | 7 | 0.14035481770833333 | 0.140126953125 |
| 20 | 8 | 20 | 275 | False | 12 | 0.1359248408564815 | 0.1350535300925926 |
| 50 | 1 | 0 | init | False | 26 | 0.16198386863425926 | 0.15726164641203705 |
| 50 | 2 | 32 | 575 | False | 11 | 0.16426396122685186 | 0.1599754050925926 |
| 50 | 3 | 65 | 1400 | False | 21 | 0.17010380497685185 | 0.16546079282407408 |
| 50 | 4 | 97 | 2200 | False | 8 | 0.1424974681712963 | 0.13950086805555556 |
| 50 | 5 | 130 | 3025 | False | 9 | 0.15116608796296296 | 0.14699327256944444 |
| 50 | 6 | 162 | 3825 | False | 65 | 0.9744737413194444 | 0.25361147280092594 |
| 50 | 7 | 195 | 4650 | False | 13 | 0.12865849247685185 | 0.12737955729166667 |
| 50 | 8 | 228 | 5475 | False | 7 | 0.1303660300925926 | 0.12861617476851853 |
| 99 | 1 | 0 | init | False | 39 | 0.126044921875 | 0.12288917824074073 |
| 99 | 2 | 5 | w124 | False | 3 | 0.12516276041666666 | 0.12506691261574074 |
| 99 | 3 | 10 | 25 | False | 9 | 0.13330656828703705 | 0.1303587962962963 |
| 99 | 4 | 15 | 150 | False | 6 | 0.11046694155092593 | 0.10925455729166667 |
| 99 | 5 | 21 | 300 | False | 6 | 0.10383897569444445 | 0.10288302951388889 |
| 99 | 6 | 26 | 425 | False | 7 | 0.1445460792824074 | 0.14276186342592592 |
| 99 | 7 | 31 | 550 | False | 5 | 0.12393952546296297 | 0.12252423321759259 |
| 99 | 8 | 37 | 700 | False | 5 | 0.10234157986111111 | 0.10175708912037038 |
| 92 | 1 | 0 | init | False | 31 | 0.12556242766203704 | 0.12223307291666667 |
| 92 | 2 | 5 | w124 | False | 3 | 0.12448929398148148 | 0.12439453125 |
| 92 | 3 | 10 | 25 | False | 10 | 0.12638057002314815 | 0.12394205729166667 |
| 92 | 4 | 15 | 150 | False | 9 | 0.11210177951388889 | 0.11076244212962963 |
| 92 | 5 | 20 | 275 | False | 6 | 0.09669958043981482 | 0.09554144965277778 |
| 92 | 6 | 25 | 400 | False | 6 | 0.12033890335648148 | 0.11899739583333334 |
| 92 | 7 | 30 | 525 | False | 13 | 0.13351671006944443 | 0.13170319733796296 |
| 92 | 8 | 35 | 650 | False | 8 | 0.11681098090277778 | 0.11557798032407407 |
| 212 | 1 | 0 | init | False | 13 | 0.1737550636574074 | 0.16762225115740742 |
| 212 | 2 | 6 | w149 | False | 4 | 0.14436378761574073 | 0.14359447337962963 |
| 212 | 3 | 13 | 100 | False | 8 | 0.15489330150462963 | 0.1524363425925926 |
| 212 | 4 | 20 | 275 | False | 24 | 0.15065538194444444 | 0.14858579282407408 |
| 212 | 5 | 26 | 425 | False | 6 | 0.15767216435185186 | 0.1549048755787037 |
| 212 | 6 | 33 | 600 | False | 11 | 0.1502220775462963 | 0.14765335648148148 |
| 212 | 7 | 40 | 775 | False | 5 | 0.13374927662037037 | 0.13237630208333334 |
| 212 | 8 | 47 | 950 | False | 10 | 0.15190719039351852 | 0.14948640046296297 |

## Run R fresh renders vs run A frames

| episode_idx | frames_compared | frames_bit_identical | max_abs_diff_max | max_abs_diff_median | mean_abs_diff_mean | share_channels_differ_median | share_channels_differ_min | share_channels_differ_max |
|---|---|---|---|---|---|---|---|---|
| 3 | 50 | 0 | 23 | 8.0 | 0.18407667100694444 | 0.1717523871527778 | 0.14492621527777777 | 0.33498444733796295 |
| 8 | 72 | 0 | 29 | 8.0 | 0.1629052432966821 | 0.15341435185185184 | 0.11062680844907408 | 0.26798972800925924 |
| 16 | 250 | 0 | 26 | 7.0 | 0.10966528790509258 | 0.10882667824074074 | 0.056910083912037036 | 0.18423213252314816 |
| 20 | 72 | 0 | 23 | 5.5 | 0.11330808537487139 | 0.10374782986111111 | 0.04638201678240741 | 0.19317744502314815 |
| 50 | 250 | 0 | 63 | 9.0 | 0.14735126880787036 | 0.13842900028935184 | 0.05245044849537037 | 0.27496925636574077 |
| 99 | 117 | 0 | 42 | 7.0 | 0.11580107183839822 | 0.1132447193287037 | 0.08474030671296297 | 0.14360930266203703 |
| 92 | 50 | 0 | 29 | 7.0 | 0.11719594907407407 | 0.11552390769675926 | 0.09506329571759259 | 0.14261574074074074 |
| 212 | 51 | 0 | 20 | 8.0 | 0.14548260768427743 | 0.14255353009259258 | 0.12819155092592593 | 0.20165762442129628 |
