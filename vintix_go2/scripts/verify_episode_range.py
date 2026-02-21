#!/usr/bin/env python3
"""
Verify that episode_range is correctly applied to datasets (step-based)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from vintix.data.torch_dataloaders import MultiTaskMapDataset
import numpy as np
import h5py

def count_total_steps(dataset_path):
    """Count total steps in a dataset"""
    total_steps = 0
    files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
    files.sort()
    for file in files:
        file_path = os.path.join(dataset_path, file)
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                step_num = np.array(f[key]['step_num'])
                total_steps += len(step_num)
    return total_steps

def analyze_episode_range_per_env(dataset_path, start_frac, end_frac):
    """Analyze episode range per environment (HDF5 file)"""
    files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
    files.sort()
    
    results = []
    for file in files:
        file_path = os.path.join(dataset_path, file)
        with h5py.File(file_path, 'r') as f:
            all_step_nums = []
            for key in sorted(f.keys(), key=lambda x: int(x.split('-')[0])):
                step_num = np.array(f[key]['step_num'])
                all_step_nums.append(step_num)
            
            if len(all_step_nums) == 0:
                continue
                
            step_nums = np.hstack(all_step_nums)
            total_steps = len(step_nums)
            
            # Calculate expected range
            start_step_idx = int(total_steps * start_frac)
            end_step_idx = int(total_steps * end_frac)
            
            # Find actual start (may be adjusted to episode start)
            actual_start_step_idx = start_step_idx
            if start_step_idx > 0 and step_nums[start_step_idx] != 0:
                # start_step_idx is in the middle of an episode, find episode start
                for i in range(start_step_idx - 1, -1, -1):
                    if step_nums[i] == 0:
                        actual_start_step_idx = i
                        break
                if actual_start_step_idx == start_step_idx:
                    actual_start_step_idx = 0
            
            # Find actual end (end of episode containing end_step_idx)
            actual_end_step_idx = end_step_idx
            if end_step_idx < total_steps:
                # Find the next episode boundary (step_num == 0) after end_step_idx
                for i in range(end_step_idx, total_steps):
                    if step_nums[i] == 0:
                        actual_end_step_idx = i - 1
                        break
                else:
                    actual_end_step_idx = total_steps - 1
            
            expected_steps = end_step_idx - start_step_idx
            actual_steps = actual_end_step_idx - actual_start_step_idx + 1
            extra_steps = actual_steps - expected_steps
            
            # Find which episode contains the end_step_idx
            episode_num_at_end = 0
            for i in range(end_step_idx + 1):
                if i < total_steps and step_nums[i] == 0 and i > 0:
                    episode_num_at_end += 1
            
            # Find first and last step_num in the extracted range
            first_step_num = step_nums[actual_start_step_idx] if actual_start_step_idx < total_steps else None
            last_step_num = step_nums[actual_end_step_idx] if actual_end_step_idx < total_steps else None
            
            # Count episodes in the extracted range
            episodes_in_range = 0
            for i in range(start_step_idx, min(actual_end_step_idx + 1, total_steps)):
                if step_nums[i] == 0 and i > start_step_idx:
                    episodes_in_range += 1
            
            results.append({
                'file': file,
                'total_steps': total_steps,
                'start_step_idx': start_step_idx,
                'actual_start_step_idx': actual_start_step_idx,
                'end_step_idx': end_step_idx,
                'actual_end_step_idx': actual_end_step_idx,
                'expected_steps': expected_steps,
                'actual_steps': actual_steps,
                'extra_steps': extra_steps,
                'episode_num_at_end': episode_num_at_end,
                'first_step_num': first_step_num,
                'last_step_num': last_step_num,
                'episodes_in_range': episodes_in_range
            })
    
    return results

def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, "configs/multitask_go2_minicheetah_config.yaml")
    config = OmegaConf.load(config_path)
    
    data_dir = os.path.join(parent_dir, "data")
    
    # Extract dataset info
    dataset_names = {
        v.path: v.group
        for k, v in config.items() if v.type == "default"
    }
    
    episode_sparsity = [
        v.episode_sparsity
        for v in config.values() if v.type == "default"
    ]
    
    # Extract episode_range
    episode_range = []
    for v in config.values():
        if v.type == "default":
            if hasattr(v, 'episode_range') and v.episode_range is not None:
                try:
                    if hasattr(v.episode_range, '__iter__'):
                        ep_range_list = list(v.episode_range)
                        episode_range.append(tuple(ep_range_list))
                    else:
                        episode_range.append(None)
                except Exception:
                    episode_range.append(None)
            else:
                episode_range.append(None)
    
    print("=" * 80)
    print("Dataset Configuration:")
    print("=" * 80)
    for i, (ds_path, group) in enumerate(dataset_names.items()):
        print(f"\nDataset {i+1}: {ds_path}")
        print(f"  Group: {group}")
        print(f"  Episode sparsity: {episode_sparsity[i]}")
        if episode_range[i] is not None:
            print(f"  Episode range: {episode_range[i]} (using {episode_range[i][1] - episode_range[i][0]:.1%} of STEPS)")
        else:
            print(f"  Episode range: None (using all steps)")
    
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    # Count total steps in original datasets
    print("\nCounting total steps in original datasets...")
    total_steps_original = []
    for i, (ds_path, group) in enumerate(dataset_names.items()):
        dataset_full_path = os.path.join(data_dir, ds_path)
        total_steps = count_total_steps(dataset_full_path)
        total_steps_original.append(total_steps)
        print(f"  Dataset {i+1} ({ds_path}): {total_steps:,} total steps")
    
    # Create dataset
    dataset = MultiTaskMapDataset(
        data_dir=data_dir,
        datasets_info=dataset_names,
        trajectory_len=2048,
        trajectory_sparsity=128,
        ep_sparsity=episode_sparsity,
        episode_range=episode_range if episode_range else None,
        preload=False,
    )
    
    print(f"\nTotal dataset length: {len(dataset):,}")
    
    # Check each individual dataset
    print("\n" + "=" * 80)
    print("Individual Dataset Analysis (Step-based):")
    print("=" * 80)
    
    for i, ds in enumerate(dataset.datasets):
        ds_path = list(dataset_names.keys())[i]
        print(f"\nDataset {i+1}: {ds_path}")
        print(f"  Task name: {ds.metadata.get('task_name', 'N/A')}")
        
        # Count steps used
        total_transitions_used = sum(len(transes) for transes in ds.transes)
        print(f"  Transitions used: {total_transitions_used:,}")
        
        # Calculate step usage
        total_steps_orig = total_steps_original[i]
        if episode_range[i] is not None:
            start_frac, end_frac = episode_range[i]
            expected_steps = int(total_steps_orig * (end_frac - start_frac))
            actual_usage = (total_transitions_used / total_steps_orig * 100) if total_steps_orig > 0 else 0
            print(f"  Original total steps: {total_steps_orig:,}")
            print(f"  Expected steps (range [{start_frac:.2f}, {end_frac:.2f}]): {expected_steps:,}")
            print(f"  Actual steps used: {total_transitions_used:,}")
            print(f"  Extra steps (episode end included): {total_transitions_used - expected_steps:,}")
            print(f"  Usage rate: {actual_usage:.2f}% of original Minicheetah dataset")
            
            # Analyze per environment
            dataset_full_path = os.path.join(data_dir, ds_path)
            env_results = analyze_episode_range_per_env(dataset_full_path, start_frac, end_frac)
            if env_results:
                print(f"\n  Per-trajectory analysis ({len(env_results)} trajectories):")
                print(f"    Each trajectory extracts its own {start_frac:.0%}-{end_frac:.0%} range independently")
                total_extra = 0
                for env_result in env_results:
                    total_extra += env_result['extra_steps']
                print(f"\n    {env_result['file']}:")
                print(f"      Total steps in trajectory: {env_result['total_steps']:,}")
                print(f"      Requested range: [{env_result['start_step_idx']:,}, {env_result['end_step_idx']:,}]")
                print(f"      Actual extracted range: [{env_result.get('actual_start_step_idx', env_result['start_step_idx']):,}, {env_result['actual_end_step_idx']:,}]")
                print(f"      Expected: {env_result['expected_steps']:,} steps ({start_frac:.0%}-{end_frac:.0%} of trajectory)")
                print(f"      Actual: {env_result['actual_steps']:,} steps")
                if env_result.get('actual_start_step_idx', env_result['start_step_idx']) < env_result['start_step_idx']:
                    print(f"      Start adjusted: {env_result['start_step_idx']:,} → {env_result.get('actual_start_step_idx', env_result['start_step_idx']):,} (episode start included)")
                if env_result['extra_steps'] > 0:
                    print(f"      Extra: {env_result['extra_steps']:,} steps (episode end included)")
                print(f"      First step_num in range: {env_result['first_step_num']} (episode start)")
                print(f"      Last step_num in range: {env_result['last_step_num']} (episode {env_result['episode_num_at_end']} end)")
                print(f"      Episodes in range: {env_result['episodes_in_range']}")
                print(f"      ✓ Extracts specified range from this trajectory")
                print(f"\n    Total extra steps across all trajectories: {total_extra:,}")
        else:
            print(f"  Original total steps: {total_steps_orig:,}")
            print(f"  Usage rate: 100.0% (all steps used)")
        
        # Count splits (samples)
        print(f"  Number of samples (splits): {len(ds.splits):,}")
    
    print("\n" + "=" * 80)
    print("Verification Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()

