import sys
from collections import defaultdict

import rospy
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

rospy.init_node('pubpose')
counter = defaultdict(int)
L = []
with open(sys.argv[1]) as f:
    for line in f:
        sign, tx, ty, tz, qx, qy, qz, qw = line.strip().split('\t')
        trans = TransformStamped()
        trans.header.frame_id = 'map'
        trans.child_frame_id = '%s%d' % (sign, counter[sign])
        counter[sign] += 1
        trans.transform.translation.x = float(tx)
        trans.transform.translation.y = float(ty)
        trans.transform.translation.z = float(tz)
        trans.transform.rotation.x = float(qx)
        trans.transform.rotation.y = float(qy)
        trans.transform.rotation.z = float(qz)
        trans.transform.rotation.w = float(qw)
        L.append(trans)

stb = StaticTransformBroadcaster()
stb.sendTransform(L)

print(L)

rospy.spin()
