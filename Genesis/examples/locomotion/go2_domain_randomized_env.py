import numpy as np
import torch
import random
from go2_env import Go2Env


class Go2DomainRandomizedEnv(Go2Env):
    """
    Go2環境にドメインランダマイゼーションを追加
    ランダマイズする要素:
    1. 質量 (mass)
    2. 摩擦係数 (friction)
    3. センサノイズ (sensor noise)
    """
    
    def __init__(self, 
                 domain_randomization=True,
                 mass_range=(0.9, 1.1),  # 質量の範囲 (スケール) - 保守的開始
                 randomize_base_only=True,  # 胴体のみランダマイズするか
                 sensor_noise_std=0.005,  # センサノイズの標準偏差（軽微なノイズ）
                 enable_sensor_noise=True,  # センサノイズを有効にするか
                 **kwargs):
        """
        Args:
            domain_randomization: ドメインランダマイゼーションを有効にするか
            mass_range: 質量のランダマイゼーション範囲 (min, max)
            randomize_base_only: 胴体のみランダマイズするか
            sensor_noise_std: センサノイズの標準偏差
            enable_sensor_noise: センサノイズを有効にするか
        """
        super().__init__(**kwargs)
        
        self.domain_randomization = domain_randomization
        self.mass_range = mass_range
        self.randomize_base_only = randomize_base_only
        self.sensor_noise_std = sensor_noise_std
        self.enable_sensor_noise = enable_sensor_noise
        
        # ランダマイゼーション用のパラメータ
        self.current_mass_scale = 1.0
        
        # 元の物理パラメータを保存
        self.original_mass = None
        
        # ドメインランダマイゼーション用の初期化は後で行う
    
    def _save_original_physics(self):
        """元の物理パラメータを保存"""
        try:
            # ベースリンクの元の質量を取得
            base_link = self.robot.get_link("base")
            self.original_mass = base_link.get_mass()
        except Exception as e:
            print(f"Warning: Could not get original mass: {e}")
            self.original_mass = 6.921  # デフォルト値（Go2の胴体質量）
    
    def _randomize_physics(self):
        """物理パラメータをランダマイズして実際に適用"""
        if not self.domain_randomization:
            return
            
        # 元の質量が保存されていない場合は保存
        if self.original_mass is None:
            self._save_original_physics()
            
        # 質量をランダマイズ
        mass_scale = random.uniform(*self.mass_range)
        self.current_mass_scale = mass_scale
        
        try:
            # 胴体のみランダマイズする場合
            if self.randomize_base_only:
                base_link = self.robot.get_link("base")
                new_mass = self.original_mass * self.current_mass_scale
                base_link.set_mass(new_mass)
                # デバッグ出力（必要に応じてコメントアウト）
                # print(f"Domain randomization: mass_scale={self.current_mass_scale:.3f}, new_mass={new_mass:.3f}kg")
            else:
                # 全リンクをランダマイズする場合（将来の拡張用）
                base_link = self.robot.get_link("base")
                new_mass = self.original_mass * self.current_mass_scale
                base_link.set_mass(new_mass)
                # print(f"Domain randomization: mass_scale={self.current_mass_scale:.3f}, new_mass={new_mass:.3f}kg")
        except Exception as e:
            print(f"Warning: Could not randomize physics: {e}")
            # エラーが発生しても学習を継続
    
    def _add_sensor_noise(self, obs):
        """観測値にセンサノイズを追加"""
        if not self.domain_randomization or not self.enable_sensor_noise or self.sensor_noise_std <= 0:
            return obs
            
        try:
            # obsがタプルの場合は最初の要素（テンソル）を取得
            if isinstance(obs, tuple):
                obs_tensor = obs[0]
            else:
                obs_tensor = obs
                
            # ノイズを追加（位置・速度・角度情報に）
            noise = torch.randn_like(obs_tensor) * self.sensor_noise_std
            noisy_obs = obs_tensor + noise
            
            # 元の形式で返す
            if isinstance(obs, tuple):
                return (noisy_obs,) + obs[1:]
            else:
                return noisy_obs
        except Exception as e:
            print(f"Warning: Could not add sensor noise: {e}")
            return obs  # エラーが発生した場合は元の観測値を返す
    
    def reset(self, **kwargs):
        """環境をリセット（物理パラメータもランダマイズ）"""
        # 物理パラメータをランダマイズ
        self._randomize_physics()
        
        # 通常のリセット処理
        obs = super().reset(**kwargs)
        
        # センサノイズを追加
        obs = self._add_sensor_noise(obs)
        
        return obs
    
    def step(self, actions):
        """ステップ実行（観測値にノイズ追加）"""
        # 通常のステップ処理
        obs, rewards, dones, infos = super().step(actions)
        
        # センサノイズを追加
        obs = self._add_sensor_noise(obs)
        
        return obs, rewards, dones, infos
    
    def get_randomization_info(self):
        """現在のランダマイゼーション情報を取得"""
        if not self.domain_randomization:
            return None
            
        return {
            'mass_scale': self.current_mass_scale,
            'sensor_noise_std': self.sensor_noise_std,
            'mass_range': self.mass_range,
            'randomize_base_only': self.randomize_base_only
        }
    
    def set_randomization_ranges(self, mass_range=None, sensor_noise_std=None):
        """ランダマイゼーション範囲を動的に変更"""
        if mass_range is not None:
            self.mass_range = mass_range
        if sensor_noise_std is not None:
            self.sensor_noise_std = sensor_noise_std


# 使用例とテスト用の関数
def test_domain_randomization():
    """ドメインランダマイゼーションのテスト"""
    import genesis as gs
    
    gs.init()
    
    # 環境設定
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0, "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0, "ang_vel": 0.25,
            "dof_pos": 1.0, "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0, "base_height": -50.0,
            "action_rate": -0.005, "similar_to_default": -0.1,
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    
        # ドメインランダマイゼーション環境を作成
    env = Go2DomainRandomizedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_randomization=True,
        mass_range=(0.9, 1.1),  # 保守的開始
        randomize_base_only=True,
        sensor_noise_std=0.01
    )
    
    print("Testing domain randomization...")
    
    # 複数回リセットしてランダマイゼーションを確認
    for i in range(5):
        obs = env.reset()
        info = env.get_randomization_info()
        print(f"Episode {i+1}: {info}")
        
        # 数ステップ実行
        for step in range(10):
            actions = torch.randn(1, 12) * 0.1
            obs, rewards, dones, infos = env.step(actions)
            
            if dones[0]:
                break
    
    print("Domain randomization test completed!")


if __name__ == "__main__":
    test_domain_randomization()
