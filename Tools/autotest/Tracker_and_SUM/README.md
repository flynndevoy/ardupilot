# Tracker_and_SUM

## Tracker selection

The ArduPlane encounter simulator uses the legacy tracker by default. To try
the sibling `tracker-source-v3` implementation, set
`ARDUPLANE_TRACKER_IMPL=v3` before launching the simulator. The v3 source tree
must remain beside `ardupilot`, and the selected Python environment must meet
its Python 3.11+ requirement and have its dependencies installed. Set the
variable to `legacy` (or unset it) to return to the existing implementation.

## To run with new tracker
cd ~/DAA_Manoeuvre2
ARDUPLANE_TRACKER_IMPL=v3 bash encounter_sim.sh 0 1 1 0 Combined1000 EO off closest 3