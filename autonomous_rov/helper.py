import math
import numpy as np

class Helper:

    def __init__(self):
        pass        
    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def eulerang(self, phi, theta, psi):
        """  
        Generate the transformation 6x6 matrix J 
        and 3x3 matrix of j_11 and j_22
        which corresponds to eq 2.40 on p.26 (Fossen 2011)
        """

        cphi = math.cos(phi)
        sphi = math.sin(phi)
        cth = math.cos(theta)
        sth = math.sin(theta)
        cpsi = math.cos(psi)
        spsi = math.sin(psi)

        if cth == 0:
            return -1

        # corresponds to eq 2.18 on p.22 (Fossen 2011)
        r_zyx = np.array([[cpsi*cth,  -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth],
                            [spsi*cth,  cpsi*cphi+sphi*sth *
                                spsi,   -cpsi*sphi+sth*spsi*cphi],
                            [-sth,      cth*sphi,                  cth*cphi]])

        # corresponds to eq 2.28 on p.25 (Fossen 2011)
        t_zyx = np.array([[1,  sphi*sth/cth,  cphi*sth/cth],
                            [0,  cphi,          -sphi],
                            [0,  sphi/cth,      cphi/cth]])

        # corresponds to eq 2.40 on p.26 (Fossen 2011)
        j_1 = np.concatenate((r_zyx, np.zeros((3, 3))), axis=1)
        j_2 = np.concatenate((np.zeros((3, 3)), t_zyx), axis=1)
        j = np.concatenate((j_1, j_2), axis=0)

        return j, r_zyx, t_zyx