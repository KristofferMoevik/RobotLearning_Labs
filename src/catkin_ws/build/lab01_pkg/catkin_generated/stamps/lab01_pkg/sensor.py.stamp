import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np

def callback(data):
    
    measurement_noise_var = 0.005
    pendulum_measured = Vector3()
    pendulum_measured.x = data.x + np.sqrt(measurement_noise_var)*np.random.randn()
    rospy.loginfo("publishing: %f", pendulum_measured.x)
    pub.publish(pendulum_measured)

def sensor():
    rospy.init_node('sensor', anonymous=True)
    rospy.Subscriber("/pendulum_state", Vector3, callback)
    rospy.spin()

if __name__ == '__main__':
    pub = rospy.Publisher('pendulum_measured', Vector3, queue_size=10)
    sensor()