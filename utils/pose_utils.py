import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _left_apply_se3(R, T, tau):
    """
    Apply one left-multiplied SE(3) update on the CPU, avoiding repeated
    construction of 4x4 matrices. All arguments are assumed on the same
    device (preferably CPU).
    """
    rho = tau[:3]
    theta = tau[3:]
    R_delta = SO3_exp(theta)
    t_delta = V(theta) @ rho

    new_R = R_delta @ R
    new_T = R_delta @ T + t_delta
    return new_R, new_T


def update_pose(camera, converged_threshold=1e-5):
    # Small SE(3) ops on the CPU avoid frequent CUDA launches and syncs
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0).detach()
    tau_cpu = tau.to(device="cpu")
    R_cpu = camera.R.detach().to(device="cpu")
    T_cpu = camera.T.detach().to(device="cpu")

    # Avoid sqrt to save one sync
    tau_norm_sq = float((tau_cpu * tau_cpu).sum())
    threshold_sq = converged_threshold * converged_threshold

    new_R_cpu, new_T_cpu = _left_apply_se3(R_cpu, T_cpu, tau_cpu)

    converged = tau_norm_sq < threshold_sq
    camera.update_RT(new_R_cpu.to(device=camera.device), new_T_cpu.to(device=camera.device))

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

