# Sweep status — driver 580.173.02 — started Fri Jul 24 12:42:44 PM MDT 2026
[07-24 12:42] SWEEP RUN (re)started
[07-24 12:42] ARM START: stretchA  transform=stretch  eps 0-300  extra='none'  (have 0)
[07-26 17:10] ARM DONE: stretchA — 300/300 scored
      n_present = 300 / n_total = 300 (0 missing — scored as failures for SR/OS/SPL)
      success_over_total = 15.0%
      oracle_success_over_total = 29.0%
    term_reason: {'step_cap': 140, 'stop': 116, 'sim_done': 24, 'wall_timeout': 20}
[07-26 17:10]   pushed: Sweep arm complete: stretchA (300/300)
[07-26 17:10] ARM START: stretchB  transform=stretch  eps 0-300  extra='none'  (have 0)
[08-04 15:32] SWEEP RUN (re)started
[08-04 15:32] ARM SKIP (complete): stretchA (300/300)
[08-04 15:32] ARM START: stretchB  transform=stretch  eps 0-300  extra='none'  (have 233)
[08-05 02:39] ARM DONE: stretchB — 300/300 scored
      n_present = 300 / n_total = 300 (0 missing — scored as failures for SR/OS/SPL)
      success_over_total = 16.0%
      oracle_success_over_total = 31.7%
    term_reason: {'step_cap': 129, 'wall_timeout': 22, 'sim_done': 28, 'stop': 121}
[08-05 02:39]   pushed: Sweep arm complete: stretchB (300/300)
[08-05 02:39] ARM START: crop  transform=crop  eps 0-300  extra='none'  (have 0)
[08-07 02:50] ARM DONE: crop — 300/300 scored
      n_present = 300 / n_total = 300 (0 missing — scored as failures for SR/OS/SPL)
      success_over_total = 16.7%
      oracle_success_over_total = 24.0%
    term_reason: {'step_cap': 170, 'stop': 107, 'wall_timeout': 6, 'sim_done': 17}
