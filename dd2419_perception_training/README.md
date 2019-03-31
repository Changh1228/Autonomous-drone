## What's what

`poses.txt`: Tab-separated list of objects, their positions and orientations.
We use a similar format to TUM, that is: each line contains a single pose, the
format is `sign tx ty tz qx qy qz qw` with tabs for separation, where `tx ty
tz` is the map position of the object's center, and `qx qy qz qw` is the
rotation quaternion of the object. `sign` is one of: 

- `airport`
- `circulation_warning`
- `dangerous_curve_left`
- `dangerous_curve_right`
- `road_narrows_from_left`
- `road_narrows_from_right`
- `no_bicycle`
- `no_heavy_truck`
- `junction`
- `no_parking`
- `no_stopping_and_parking`
- `residential`
- `stop`
- `follow_left`
- `follow_right`

These are the same names as in the rest of the course.

`scripts/pubpose.py` publishes TF transforms for all signs in the given text
file. This is for illustration only.

`bags/long.bag` and `bags/short.bag` are the two main sources of data you will
use. Play them back and while running your perception system to see how you
fare.

Your results must be in the same format as the provided `poses.txt`.
