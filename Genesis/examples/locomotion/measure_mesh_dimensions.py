"""
Meshファイル（.obj）から寸法情報を計測するスクリプト
trimeshライブラリを使用してバウンディングボックスから寸法を計算
"""
import os
import sys
try:
    import trimesh
except ImportError:
    print("エラー: trimeshライブラリがインストールされていません。")
    print("インストール方法: pip install trimesh")
    sys.exit(1)

def measure_mesh_dimensions(mesh_path):
    """Meshファイルを読み込んで寸法を計測"""
    if not os.path.exists(mesh_path):
        print(f"警告: Meshファイルが見つかりません: {mesh_path}")
        return None
    
    try:
        mesh = trimesh.load(mesh_path)
        
        # バウンディングボックスを取得
        bounds = mesh.bounds
        # 各軸方向の寸法を計算
        extents = mesh.extents  # [length_x, length_y, length_z]
        
        # バウンディングボックスの最小値と最大値
        min_bounds = bounds[0]
        max_bounds = bounds[1]
        
        # 寸法情報を返す
        # URDFでは通常: length (x), width (y), height (z)
        dimensions = {
            "length": extents[0],  # x方向
            "width": extents[1],   # y方向
            "height": extents[2],  # z方向
            "extents": extents,
            "bounds": bounds,
        }
        
        return dimensions
    except Exception as e:
        print(f"エラー: Meshファイルの読み込みに失敗しました: {e}")
        return None

# Mini CheetahのMeshファイルのパス
mesh_files = {
    "body": "/workspace/Genesis/genesis/assets/urdf/mini_cheetah/meshes/mini_body.obj",
    "shank": "/workspace/Genesis/genesis/assets/urdf/mini_cheetah/meshes/mini_lower_link.obj",
}

print("=" * 60)
print("Mini Cheetah Mesh寸法計測")
print("=" * 60)

results = {}

for part_name, mesh_path in mesh_files.items():
    print(f"\n{part_name.upper()} ({mesh_path}):")
    print("-" * 60)
    
    dims = measure_mesh_dimensions(mesh_path)
    
    if dims:
        results[part_name] = dims
        print(f"  長さ (Length, X軸): {dims['length']:.4f} m")
        print(f"  幅 (Width, Y軸):   {dims['width']:.4f} m")
        print(f"  高さ (Height, Z軸): {dims['height']:.4f} m")
        print(f"\n  バウンディングボックス:")
        print(f"    Min: {dims['bounds'][0]}")
        print(f"    Max: {dims['bounds'][1]}")
    else:
        results[part_name] = None
        print("  計測失敗")

print("\n" + "=" * 60)
print("計測結果サマリー:")
print("=" * 60)

if results.get("body"):
    body = results["body"]
    print(f"\nBase (body):")
    print(f"  Base length (m): {body['length']:.3f}")
    print(f"  Base width (m):  {body['width']:.3f}")
    print(f"  Base height (m): {body['height']:.3f}")

if results.get("shank"):
    shank = results["shank"]
    # shankの長さは、最も長い軸方向の寸法とする
    shank_length = max(shank['length'], shank['width'], shank['height'])
    print(f"\nCalf/Shank (mini_lower_link):")
    print(f"  最大寸法 (Calf length, m): {shank_length:.3f}")
    print(f"  各軸方向の寸法:")
    print(f"    X (length): {shank['length']:.3f} m")
    print(f"    Y (width):  {shank['width']:.3f} m")
    print(f"    Z (height): {shank['height']:.3f} m")

print("\n" + "=" * 60)
