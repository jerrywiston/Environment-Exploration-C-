import numpy as np
import matplotlib.pyplot as plt

class SimpleMotionModel:
    def __init__(self, normal_var, tangent_var, angular_var):
        self.normal_var = normal_var
        self.tangent_var = tangent_var
        self.angular_var = angular_var

    def Sample(self, pos_, n, t, theta):
        pos = pos_.copy()
        n = n + self.normal_var*np.random.randn()
        t = t + self.tangent_var*np.random.randn()
        th = theta + self.angular_var*np.random.randn()
        pos[2] = (pos[2] + th) % 360
        pos[0] = pos[0] + n*np.cos(np.deg2rad(pos[2])) + t*np.sin(np.deg2rad(pos[2]))
        pos[1] = pos[1] + n*np.sin(np.deg2rad(pos[2])) + t*np.cos(np.deg2rad(pos[2]))
        return pos

class VelocityMotionModel:
    def __init__(self, params, dt=1):
        self.dt = dt
        self.params = params #[a1, a2, a3, a4, a5, a6]
    
    def Sample(self, pos, v, w):
        v_ = v + np.random.randn() * (self.params[0]*np.abs(v) + self.params[1]*np.abs(w))
        w_ = w + np.random.randn() * (self.params[2]*np.abs(v) + self.params[3]*np.abs(w))
        g_ = np.random.randn() * (self.params[4]*np.abs(v) + self.params[5]*np.abs(w))
        
        if w_ == 0:
            w_ = 1e-5

        x_ = pos[0] - (v_/w_) * (np.sin(np.deg2rad(pos[2])) + np.sin(np.deg2rad(pos[2] + w_*self.dt)))
        y_ = pos[1] + (v_/w_) * (np.cos(np.deg2rad(pos[2])) - np.cos(np.deg2rad(pos[2] + w_*self.dt)))
        theta_ = pos[2] + w_*self.dt + v_*self.dt
        
        return [x_, y_, theta_]

class OdometryMotionModel:
    pass

if __name__ == '__main__':
    # Simple Motion Model Test
    '''
    m_model = SimpleMotionModel(0.1, 0.1, 3)
    a = np.zeros((500,3))
    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "b.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "g.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 1, 0, 0)
        plt.plot(a[i,0], a[i,1], "r.")

    plt.axis('equal')
    plt.show()
    '''
    
    # Velocity Motion Model Test
    m_model = VelocityMotionModel([0.01, 0.01, 0.1, 0.1, 0.01, 0.01], 0.01)
    a = np.zeros((200, 3))

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 6, 100)
        plt.plot(a[i,0], a[i,1], "b.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 6, 100)
        plt.plot(a[i,0], a[i,1], "g.")

    for i in range(a.shape[0]):
        a[i] = m_model.Sample(a[i], 6, 100)
        plt.plot(a[i,0], a[i,1], "r.")

    plt.axis('equal')
    plt.show()
