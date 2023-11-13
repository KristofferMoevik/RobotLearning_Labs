#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np
from scipy.integrate import odeint
import time

def get_pendulum_state(x, dt):
    g=9.81
    l=1
    dxdt=np.array([x[1], -(g/l)*np.sin(x[0])])
    return x + 0.1*dxdt

def pendulum():
    pub = rospy.Publisher('pendulum_state', Vector3, queue_size=10)
    rospy.init_node('pendulum', anonymous=True)
    rate = 10 # 10hz
    node_rate = rospy.Rate(rate) 
    
    # Discretization time step
    deltaTime = 0.1

    # Initial true state
    x0 = np.array([np.pi/3, 0.5])

    # System dynamics
    def stateSpaceModel(x, t):
        g = 9.81
        l = 1
        dxdt = np.array([x[1], -(g/l) * np.sin(x[0])])
        return dxdt

    # Start time
    startTime = 0
    x_t = x0

    while not rospy.is_shutdown():
        # Update the state
        t = np.array([startTime, startTime + deltaTime])
        x_t = odeint(stateSpaceModel, x_t, t)[-1]

        # Handle timing (real-time or simulation time)
        startTime += deltaTime
        
        vector = Vector3()
        vector.x = x_t[0]
        vector.y = x_t[1]

        rospy.loginfo("theta: %f, theta_dot: %f", x_t[0], x_t[1])
        pub.publish(vector)
        node_rate.sleep()

if __name__ == '__main__':
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass
