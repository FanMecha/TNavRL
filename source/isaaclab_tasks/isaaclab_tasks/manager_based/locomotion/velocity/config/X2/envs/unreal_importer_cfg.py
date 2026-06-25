# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .unreal_importer import UnRealImporter


@configclass
class UnRealImporterCfg(TerrainImporterCfg):
    class_type: type = UnRealImporter
    """The class name of the terrain importer."""

    people_config_file: str | None = None
