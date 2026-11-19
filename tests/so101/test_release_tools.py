"""Hardware-free checks for the migrated synchronous policy runner."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('so101_run_policy', ROOT / 'scripts/so101/run_policy.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class RunnerTests(unittest.TestCase):
    def test_camera_and_joint_contract(self):
        feature = types.SimpleNamespace(shape=(3, 480, 640))
        cfg = types.SimpleNamespace(image_features={'observation.images.fixed': feature},
                                    input_features=['observation.images.fixed', 'observation.state'])
        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        observation = {f'{name}.pos': i for i, name in enumerate(names)}
        observation['observation.images.fixed'] = np.full((480, 640, 3), 255, dtype=np.uint8)
        batch = runner.observation_to_batch(None, cfg, observation, 'task', 'cpu')
        self.assertEqual(tuple(batch['observation.images.fixed'].shape), (1, 3, 480, 640))
        self.assertEqual(batch['observation.images.fixed'].max().item(), 1)
        self.assertEqual(batch['observation.state'].tolist(), [[0, 1, 2, 3, 4, 5]])
        observation['fixed'] = observation.pop('observation.images.fixed')
        raw_batch = runner.observation_to_batch(None, cfg, observation, 'task', 'cpu')
        self.assertTrue(torch.equal(raw_batch['observation.images.fixed'], batch['observation.images.fixed']))
        observation['observation.images.fixed'] = observation.pop('fixed')
        observation['observation.images.fixed'] = np.zeros((240, 320, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            runner.observation_to_batch(None, cfg, observation, 'task', 'cpu')

    def test_single_chunk_generation_and_no_dry_targets(self):
        class Robot:
            connected = False
            disconnected = False
            sent = 0
            def connect(self): self.connected = True
            def get_observation(self): return {}
            def send_action(self, goal): self.sent += 1; return goal
            def disconnect(self): self.disconnected = True
        class Policy:
            calls = 0
            generated = 0
            def reset(self): self.calls = 0
            def select_action(self, batch):
                if self.calls % 3 == 0: self.generated += 1
                self.calls += 1
                return torch.zeros(1, 6)
            def predict_action_chunk(self, batch):
                raise AssertionError('select_action already handles queue refill')
        robot, policy = Robot(), Policy()
        args = types.SimpleNamespace(execute=False, robot_port='EXAMPLE', robot_id='test',
                                     seconds=.045, snapshot_dir=None, device='cpu', fps=30, task='test')
        config = types.SimpleNamespace(n_action_steps=3)
        tick = iter(i * .001 for i in range(1000))
        result = {}
        with patch.object(runner, 'make_robot', return_value=robot), \
             patch.object(runner, 'observation_to_batch', return_value={}), \
             patch.object(runner.time, 'perf_counter', side_effect=lambda: next(tick)), \
             patch('lerobot.utils.robot_utils.busy_wait', return_value=None):
            runner.run_live(args, policy, config, result)
        self.assertTrue(robot.connected and robot.disconnected)
        self.assertEqual(robot.sent, 0)
        self.assertEqual(result['chunks_predicted'], policy.generated)
        self.assertGreater(policy.generated, 1)

if __name__ == '__main__':
    unittest.main()
