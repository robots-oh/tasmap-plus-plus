import numpy as np
import math
import torch

from .macros import tp

def euler_to_rot_matrix(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    if extrinsic:
        yaw, pitch, roll = euler_angles
    else:
        roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    if extrinsic:
        return np.array(
            [
                [(cp * cr), ((cr * sp * sy) - (cy * sr)), ((cr * cy * sp) + (sr * sy))],
                [(cp * sr), ((cy * cr) + (sr * sp * sy)), ((cy * sp * sr) - (cr * sy))],
                [-sp, (cp * sy), (cy * cp)],
            ]
        )
    else:
        return np.array(
            [
                [(cp * cy), (-cp * sy), sp],
                [((cy * sr * sp) + (cr * sy)), ((cr * cy) - (sr * sp * sy)), (-cp * sr)],
                [((-cr * cy * sp) + (sr * sy)), ((cy * sr) + (cr * sp * sy)), (cr * cp)],
            ]
        )

def quaternion_rotation_matrix(Q, torch_=False):
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    if torch_:
        rot_matrix = torch.tensor([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]], device=tp.DEVICE)
    else:        
        rot_matrix = np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]], dtype=float)
                                
    return rot_matrix

def rot_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (3, 3):
        tmp = np.eye(4)
        tmp[0:3, 0:3] = mat
        mat = tmp

    q = np.empty((4,), dtype=np.float64)
    t = np.trace(mat)
    if t > mat[3, 3]:
        q[0] = t
        q[3] = mat[1, 0] - mat[0, 1]
        q[2] = mat[0, 2] - mat[2, 0]
        q[1] = mat[2, 1] - mat[1, 2]
    else:
        i, j, k = 0, 1, 2
        if mat[1, 1] > mat[0, 0]:
            i, j, k = 1, 2, 0
        if mat[2, 2] > mat[i, i]:
            i, j, k = 2, 0, 1
        t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
        q[i + 1] = t
        q[j + 1] = mat[i, j] + mat[j, i]
        q[k + 1] = mat[k, i] + mat[i, k]
        q[0] = mat[k, j] - mat[j, k]
    q *= 0.5 / np.sqrt(t * mat[3, 3])
    return q
    
def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    mat = np.array(euler_to_rot_matrix(euler_angles, degrees=degrees, extrinsic=extrinsic))
    return rot_matrix_to_quat(mat)
