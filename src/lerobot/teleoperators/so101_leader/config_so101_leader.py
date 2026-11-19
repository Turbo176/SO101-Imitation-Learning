#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader")
@dataclass
class SO101LeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    # Number of additional attempts for position reads. A small amount of packet loss is common on
    # USB-to-serial adapters, especially when cameras share the same USB hub.
    position_read_num_retry: int = 3

    # Number of additional attempts for acknowledged writes during startup configuration.
    configuration_num_retry: int = 3

    use_degrees: bool = False

    def __post_init__(self):
        if self.position_read_num_retry < 0:
            raise ValueError("position_read_num_retry must be greater than or equal to 0")
        if self.configuration_num_retry < 0:
            raise ValueError("configuration_num_retry must be greater than or equal to 0")
