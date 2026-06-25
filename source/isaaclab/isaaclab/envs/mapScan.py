from __future__ import annotations  # 必须放在第一行！允许在运行时推迟类型解析
import torch
import carb
from typing import TYPE_CHECKING  # 引入类型检查开关
from omni.physx import get_physx_scene_query_interface

# ★★★ 核心修改：只在类型检查阶段导入 ManagerBasedEnv ★★★
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class MapScanner:
    """
    用于扫描 USD 地形并缓存所有可行走点（Valid Points）的静态工具类。
    """

    @staticmethod
    def get_or_create_cache(env: ManagerBasedEnv, ranges: dict, step: float = 0.5, check_radius: float = 0.5,
                            height: float = 1.0, debug_name: str = "default"):
        """
        检查 env 中是否已有缓存，没有则生成。
        """
        cache_name = f"pose_cache_{debug_name}"

        # 如果缓存已存在，直接返回
        if hasattr(env, cache_name):
            return getattr(env, cache_name)

        print(f"[MapScanner] Generating valid point cache for '{debug_name}'...")
        print(f"[MapScanner] Range: X{ranges['x']}, Y{ranges['y']}. Step: {step}m. This may take a few seconds.")

        physx = get_physx_scene_query_interface()
        points = []

        x_min, x_max = ranges['x']
        y_min, y_max = ranges['y']

        # 开始扫描网格
        curr_x = x_min
        while curr_x <= x_max:
            curr_y = y_min
            while curr_y <= y_max:
                # 碰撞检测
                check_pos = carb.Float3(curr_x, curr_y, height)
                is_colliding = physx.overlap_sphere(
                    check_radius, check_pos, lambda hit: True, False
                )

                if not is_colliding:
                    # 存入列表: [x, y, z]
                    points.append([curr_x, curr_y, height])

                curr_y += step
            curr_x += step

        if len(points) == 0:
            raise RuntimeError(f"[MapScanner] No valid points found for '{debug_name}'! Check your ranges and height.")

        # 转为 GPU Tensor
        valid_points_tensor = torch.tensor(points, device=env.device, dtype=torch.float32)

        # 存入 env 防止被回收，并方便下次调用
        setattr(env, cache_name, valid_points_tensor)

        print(f"[MapScanner] Cache '{debug_name}' generated! Found {len(points)} valid spots.")
        return valid_points_tensor