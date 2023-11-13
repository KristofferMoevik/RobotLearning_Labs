import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np


class ExtendedKalmanFilter(object):
    
    
    def __init__(self, x0, P0, Q, R, dT):
        """
           Initialize EKF
            
            Parameters
            x0 - mean of initial state prior
            P0 - covariance of initial state prior
            Q  - covariance matrix of the process noise 
            R  - covariance matrix of the measurement noise
            dT - discretization step for forward dynamics
        """
        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT
        
        
        self.g = 9.81  # Gravitational constant
        self.l = 1  # Length of the pendulum 
        
        self.currentTimeStep = 0
        

        self.priorMeans = []
        self.priorMeans.append(None)  # no prediction step for timestep=0
        self.posteriorMeans = []
        self.posteriorMeans.append(x0)
        
        self.priorCovariances=[]
        self.priorCovariances.append(None)  # no prediction step for timestep=0
        self.posteriorCovariances=[]
        self.posteriorCovariances.append(P0)
    
    

    def stateSpaceModel(self, x, t):
        """
            Dynamics may be described as a system of first-order
            differential equations: 
            dx(t)/dt = f(t, x(t))

            Dynamics are time-invariant in our case, so t is not used.
            
            Parameters:
                x : state variables (column-vector)
                t : time

            Returns:
                f : dx(t)/dt, describes the system of ODEs
        """
        dxdt = np.array([[x[1,0]], [-(self.g/self.l)*np.sin(x[0,0])]])
        return dxdt
    
    def discreteTimeDynamics(self, x_t):
        """
            Forward Euler integration.
            
            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """
        x_tp1 = x_t + self.dT*self.stateSpaceModel(x_t, None)
        return x_tp1
    

    def jacobianStateEquation(self, x_t):
        """
            Jacobian of discrete dynamics w.r.t. the state variables,
            evaluated at x_t

            Parameters:
                x_t : state variables (column-vector)
        """
        A = np.zeros(shape=(2,2))  # TODO: shape?
        
        # TODO
        # compute the Jacobian of the discrete dynamics
        # ...
        A[0,0] = 1
        A[0,1] = self.dT
        A[1,0] = -self.dT*(self.g/self.l)*np.cos(x_t[0,0])
        A[1,1] = 1

        return A
    
    
    def jacobianMeasurementEquation(self, x_t):
        """
            Jacobian of measurement model.

            Measurement model is linear, hence its Jacobian
            does not actually depend on x_t
        """
        C = np.zeros(shape=(1,2))  # TODO: shape?
        
        # TODO
        # compute the Jacobian of the measurement model
        # ...
        C[0,0] = 1
        C[0,1] = 0

        return C
    
     
    def forwardDynamics(self):
        self.currentTimeStep = self.currentTimeStep+1  # t-1 ---> t

        
        """
            Predict the new prior mean for timestep t
        """
        x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        

        """
            Predict the new prior covariance for timestep t
        """
        # Linearization: jacobian of the dynamics at the current a posteriori estimate
        A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1])

        # TODO: propagate the covariance matrix forward in time
        x_t_prior_cov = A_t_minus @ self.posteriorCovariances[self.currentTimeStep-1] @ np.transpose(A_t_minus) + self.Q
        
        # Save values
        self.priorMeans.append(x_t_prior_mean)
        self.priorCovariances.append(x_t_prior_cov)
    

    def updateEstimate(self, z_t):
        """
            Compute Posterior Gaussian distribution,
            given the new measurement z_t
        """

        # Jacobian of measurement model at x_t
        Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep]) 
        
        # TODO: Compute the Kalman gain matrix
        Cov_t = self.priorCovariances[self.currentTimeStep]
        K_t = Cov_t@np.transpose(Ct)@np.linalg.inv(Ct@Cov_t@np.transpose(Ct) + self.R)
        
        # TODO: Compute posterior mean
        x_t_prior_mean = self.priorMeans[self.currentTimeStep]
        x_t_mean = x_t_prior_mean + K_t@(z_t - Ct@x_t_prior_mean)
        
        # TODO: Compute posterior covariance
        x_t_cov = (np.identity(2) - K_t*Ct) @ Cov_t
        
        # Save values
        self.posteriorMeans.append(x_t_mean)
        self.posteriorCovariances.append(x_t_cov)


def callback(data):
    # PREDICT step
    EKF.forwardDynamics()
    
    EKF.updateEstimate(data.x)

    pendulum_ekf = Vector3()
    pendulum_ekf.x = EKF.posteriorMeans[-1][0]
    pendulum_ekf.y = EKF.posteriorMeans[-1][1]

    rospy.loginfo("theta: " + str(pendulum_ekf.x) + " theta_dot: " + str(pendulum_ekf.y))

    pub.publish(pendulum_ekf)

def ekf():

    rospy.init_node('ekf', anonymous=True)
    rospy.Subscriber("/pendulum_measured", Vector3, callback)
    rospy.spin()

if __name__ == '__main__':
    pub = rospy.Publisher('pendulum_ekf', Vector3, queue_size=10)
    """
    EKF initialization
    """
    deltaTime = 0.1
    x0 = np.array([np.pi/3, 0.5])
    # Initial state belief distribution (EKF assumes Gaussian distributions)
    x_0_mean = np.zeros(shape=(2,1))  # column-vector
    x_0_mean[0] = x0[0] + 3*np.random.randn()
    x_0_mean[1] = x0[1] + 3*np.random.randn()
    x_0_cov = 10*np.eye(2,2)  # initial value of the covariance matrix

    # Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
    Q=0.00001*np.eye(2,2)

    # Measurement noise covariance matrix for EKF
    R = np.array([[0.00005]])

    # create the extended Kalman filter object
    EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)


    ekf()