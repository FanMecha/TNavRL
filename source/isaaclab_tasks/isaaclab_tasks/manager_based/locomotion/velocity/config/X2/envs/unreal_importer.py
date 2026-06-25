# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import carb
import numpy as np
import omni
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import trimesh
import yaml
import random, math, time

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
import omni.kit.app
import omni.usd
import omni.physx as physx

if TYPE_CHECKING:
    from .unreal_importer_cfg import UnRealImporterCfg


class UnRealImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: UnRealImporterCfg

    def __init__(self, cfg: UnRealImporterCfg) -> None:
        """
        :param
        """
        super().__init__(cfg)

        if self.cfg.people_config_file:
            self._insert_people()


    """
    Import Functions
    """

    def _insert_people(self):
        # load people config file
        with open(self.cfg.people_config_file) as stream:
            people_cfg: dict = yaml.safe_load(stream)

        # if self.cfg.scale == 1.0:
        #     scale_people = 100
        # else:
        #     scale_people = 1

        for key, person_cfg in people_cfg.items():
            carb.log_verbose(f"Insert person '{key}'")

            self.insert_single_person(
                person_cfg["prim_name"],
                person_cfg["translation"],
                scale_people=1,
                usd_path=person_cfg.get("usd_path", "People/Characters/F_Business_02/F_Business_02.usd"),
            )
            # TODO: movement of the people

        carb.log_info(f"Number of people added: {len(people_cfg)}")
        print(f"Number of people added: {len(people_cfg)}")

        return

    def insert_single_person(
            self,
            prim_name: str,
            translation: list,
            scale_people: float = 1.0,
            usd_path: str = "People/Characters/F_Business_02/F_Business_02.usd",
    ) -> None:
        person_prim = prim_utils.create_prim(
            prim_path=os.path.join("/World/ground/", prim_name),
            translation=tuple(translation),
            usd_path=usd_path,
            scale=(scale_people, scale_people, scale_people),
        )

        self.randomize_orientation(person_prim)
        UsdGeom.Mesh(person_prim)
        return

    def randomize_orientation(self, person_prim):
        yaw_rad = math.radians(random.uniform(-180.0, 180.0))
        quat = Gf.Quatd(math.cos(yaw_rad * 0.5), Gf.Vec3d(0.0, 0.0, math.sin(yaw_rad * 0.5)))
        attr = person_prim.GetAttribute("xformOp:orient")
        if attr:
            if isinstance(attr.Get(), Gf.Quatf):
                attr.Set(Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(quat.GetImaginary())))
            else:
                attr.Set(quat)