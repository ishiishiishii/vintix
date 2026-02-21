"""
URDFファイルを解析して、Genloco論文スタイルのロボット仕様表を作成するスクリプト
urdf_parser_pyライブラリを使用して正確に情報を抽出

注意: 
- Box/Cylinder/Sphereなどのprimitive geometryを使用している場合は、URDFから直接サイズ情報を取得できます
- Meshファイル（.obj, .stl, .daeなど）を使用している場合、URDFには寸法情報が含まれていないため、
  以下の方法で取得する必要があります：
  1. Blenderなどの3DモデリングソフトでMeshファイルを開き、手動で計測する
  2. Meshファイルをプログラムで解析する（trimeshなどを使用）
  
現在の実装では、Meshを使用している場合、該当する値は「-」と表示されます。
例: Mini Cheetahのbody（base）とshank（calf）はMeshを使用しているため、取得できません。
"""
import os
import sys
from pathlib import Path

# スクリプトの位置からURDFパスを解決（Docker/ローカル両対応）
_SCRIPT_DIR = Path(__file__).resolve().parent
_GENESIS_ROOT = _SCRIPT_DIR.parent.parent  # Genesis/examples/locomotion -> Genesis
_URDF_BASE = _GENESIS_ROOT / "genesis" / "assets" / "urdf"
try:
    import urdf_parser_py.urdf as urdf
    USE_URDF_PARSER = True
except ImportError:
    print("警告: urdf_parser_pyがインストールされていません。xml.etree.ElementTreeを使用します。")
    import xml.etree.ElementTree as ET
    USE_URDF_PARSER = False

# ロボットの設定
# URDFファイルから直接取得できる情報のみを使用
ROBOT_CONFIGS = {
    "A1": {
        "urdf_path": str(_URDF_BASE / "unitree_a1" / "urdf" / "a1.urdf"),
        "base_link_name": "trunk",
    },
    "Go1": {
        "urdf_path": str(_URDF_BASE / "go1" / "urdf" / "go1.urdf"),
        "base_link_name": "trunk",
    },
    "Go2": {
        "urdf_path": str(_URDF_BASE / "go2" / "urdf" / "go2.urdf"),
        "base_link_name": "base",
    },
    "Mini Cheetah": {
        "urdf_path": str(_URDF_BASE / "mini_cheetah" / "urdf" / "mini_cheetah.urdf"),
        "base_link_name": "body",
        # Meshファイルから計測した値（trimeshを使用）
        "mesh_measurements": {
            "base_length": 0.276,  # mini_body.objから計測
            "base_width": 0.197,   # mini_body.objから計測
            "base_height": 0.099,  # mini_body.objから計測
            "calf_length": 0.215,  # mini_lower_link.objから計測（Z軸方向）
        },
    },
    "ANYmalC": {
        "urdf_path": str(_URDF_BASE / "anymal_c" / "urdf" / "anymal_c.urdf"),
        "base_link_name": "base",
    },
    "SpotMicro": {
        "urdf_path": str(_URDF_BASE / "spotmicro" / "urdf" / "spotmicro.urdf"),
        "base_link_name": "base_link",
    },
}

def get_geometry_size(geometry):
    """geometryオブジェクトからサイズ情報を取得"""
    if geometry is None:
        return None
    
    # urdf_parser_pyでは、geometryが直接Box, Cylinder, Sphereオブジェクトの場合がある
    if isinstance(geometry, urdf.Box):
        return list(geometry.size)
    elif isinstance(geometry, urdf.Cylinder):
        # cylinder: 直径×直径×高さ (radius*2 を x,y に、length を z に)
        d = geometry.radius * 2
        return [d, d, geometry.length]
    elif isinstance(geometry, urdf.Sphere):
        radius_2 = geometry.radius * 2
        return [radius_2, radius_2, radius_2]
    
    # 属性としてアクセスする場合
    if hasattr(geometry, 'box') and geometry.box is not None:
        return list(geometry.box.size)
    elif hasattr(geometry, 'cylinder') and geometry.cylinder is not None:
        c = geometry.cylinder
        d = c.radius * 2
        return [d, d, c.length]
    elif hasattr(geometry, 'sphere') and geometry.sphere is not None:
        radius_2 = geometry.sphere.radius * 2
        return [radius_2, radius_2, radius_2]
    
    return None

def parse_urdf_specs_urdf_parser(urdf_path, base_link_name, mesh_measurements=None):
    """urdf_parser_pyを使用してURDFファイルを解析"""
    try:
        robot = urdf.URDF.from_xml_file(urdf_path)
    except Exception as e:
        print(f"警告: URDFファイルの解析に失敗しました: {e}")
        return None
    
    specs = {
        "total_weight": 0.0,
        "base_length": 0.0,
        "base_width": 0.0,
        "base_height": 0.0,
        "thigh_length": 0.0,
        "calf_length": 0.0,
        "height_fully_standing": 0.0,
    }
    
    # 1. 総重量の計算
    for link in robot.links:
        if link.inertial and link.inertial.mass:
            specs["total_weight"] += link.inertial.mass
    
    # 2. ベースリンクのサイズを取得
    # まずbase_link_nameで探す、見つからない場合は"trunk","base","body","base_link"も試す
    base_link = None
    for name in [base_link_name, "trunk", "base", "body", "base_link"]:
        base_link = next((link for link in robot.links if link.name == name), None)
        if base_link:
            break
    
    if base_link:
        # collision geometryを優先的に使用
        if base_link.collisions:
            for collision in base_link.collisions:
                if collision.geometry:
                    size = get_geometry_size(collision.geometry)
                    if size and len(size) >= 3:
                        specs["base_length"] = max(specs["base_length"], size[0])
                        specs["base_width"] = max(specs["base_width"], size[1])
                        specs["base_height"] = max(specs["base_height"], size[2])
        
        # collisionがない場合はvisual geometryを使用
        if specs["base_length"] == 0.0 and base_link.visuals:
            for visual in base_link.visuals:
                if visual.geometry:
                    size = get_geometry_size(visual.geometry)
                    if size and len(size) >= 3:
                        specs["base_length"] = max(specs["base_length"], size[0])
                        specs["base_width"] = max(specs["base_width"], size[1])
                        specs["base_height"] = max(specs["base_height"], size[2])
    
    # 3. 太ももとすねのリンクを探索
    # ANYmalC: LF_THIGH, LF_thigh_fixed / LF_SHANK, LF_shank_fixed
    # SpotMicro: front_left_leg_link (thigh), front_left_foot_link (calf)
    thigh_links = [
        link for link in robot.links
        if ('thigh' in link.name.lower() or 'upper' in link.name.lower() or
            ('leg_link' in link.name.lower() and 'cover' not in link.name.lower() and 'foot' not in link.name.lower()))
    ]
    calf_links = [
        link for link in robot.links
        if ('calf' in link.name.lower() or 'lower' in link.name.lower() or
            'shank' in link.name.lower() or
            ('foot_link' in link.name.lower() and 'toe' not in link.name.lower()))
    ]
    
    # 4. ジョイントから脚の長さを計算
    for joint in robot.joints:
        if not joint.origin:
            continue
        
        xyz = joint.origin.xyz if hasattr(joint.origin, 'xyz') else [0, 0, 0]
        z_distance = abs(xyz[2])
        total_distance = (xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5
        
        joint_name_lower = joint.name.lower()
        child_link_name = joint.child
        parent_link_name = joint.parent
        
        # 太ももの長さ: thighからcalfへのジョイントの距離
        if "calf_joint" in joint_name_lower:
            child_link = next((link for link in robot.links if link.name == child_link_name), None)
            if child_link and any('calf' in child_link.name.lower() or 'shank' in child_link.name.lower() for _ in [1]):
                if z_distance > 0.05:
                    specs["thigh_length"] = max(specs["thigh_length"], z_distance)
                elif total_distance > 0.05:
                    specs["thigh_length"] = max(specs["thigh_length"], total_distance)
        
        # Mini Cheetah用: thigh_to_knee joint (thigh length)
        # ANYmalC用: LF_thigh_fixed_LF_KFE (KFE=knee)
        if "thigh" in joint_name_lower and ("knee" in joint_name_lower or "kfe" in joint_name_lower):
            if z_distance > 0.05:
                specs["thigh_length"] = max(specs["thigh_length"], z_distance)
            elif total_distance > 0.05:
                specs["thigh_length"] = max(specs["thigh_length"], total_distance)
        
        # すねの長さ: calf/shankからfoot/toeへのジョイントの距離
        # ANYmalC: LF_shank_fixed_LF_FOOT / SpotMicro: foot_motor_front_left (leg→foot)
        if ("foot" in joint_name_lower or "toe" in joint_name_lower):
            parent_link = next((link for link in robot.links if link.name == parent_link_name), None)
            if parent_link:
                pn = parent_link.name.lower()
                is_calf_parent = ('calf' in pn or 'shank' in pn or
                                  ('leg_link' in pn and 'cover' not in pn and 'foot' not in pn))
                if is_calf_parent:
                    if z_distance > 0.05:
                        specs["calf_length"] = max(specs["calf_length"], z_distance)
                    elif total_distance > 0.05:
                        specs["calf_length"] = max(specs["calf_length"], total_distance)
        
        # Mini Cheetah用: shankリンクが末端の場合、inertial originから推定を試みる
        # ただし、これは正確ではない可能性があるため、最後の手段として使用
        if "shank" in parent_link_name.lower() and "shank" in child_link_name.lower():
            # shankリンクのinertial originを確認（リンクの中心位置を示す可能性）
            # ただし、これはcalf lengthの正確な値を保証しない
            pass  # 現時点では使用しない（Meshから取得できないため）
    
    # 5. リンクのcollision geometryから長さを取得（補完的に）
    for link in thigh_links:
        if link.collisions:
            for collision in link.collisions:
                if collision.geometry:
                    size = get_geometry_size(collision.geometry)
                    if size:
                        # boxまたはcylinderの場合、最大サイズを長さとする
                        max_size = max([s for s in size if s > 0]) if size else 0
                        if max_size > specs["thigh_length"]:
                            specs["thigh_length"] = max_size
    
    for link in calf_links:
        if link.collisions:
            for collision in link.collisions:
                if collision.geometry:
                    size = get_geometry_size(collision.geometry)
                    if size:
                        # boxまたはcylinderの場合、最大サイズを長さとする
                        max_size = max([s for s in size if s > 0]) if size else 0
                        if max_size > specs["calf_length"]:
                            specs["calf_length"] = max_size
    
    # Height, fully standing はURDFから直接取得できないため0に設定
    specs["height_fully_standing"] = 0.0
    
    # Meshファイルから計測した値を適用（Mini Cheetahなど）
    if mesh_measurements:
        if "base_length" in mesh_measurements and specs["base_length"] == 0.0:
            specs["base_length"] = mesh_measurements["base_length"]
        if "base_width" in mesh_measurements and specs["base_width"] == 0.0:
            specs["base_width"] = mesh_measurements["base_width"]
        if "base_height" in mesh_measurements and specs["base_height"] == 0.0:
            specs["base_height"] = mesh_measurements["base_height"]
        if "calf_length" in mesh_measurements and specs["calf_length"] == 0.0:
            specs["calf_length"] = mesh_measurements["calf_length"]
    
    return specs

def parse_urdf_specs_xml(urdf_path, base_link_name, thigh_joint_patterns, calf_joint_patterns, known_values=None):
    """XMLパーサーを使用したフォールバック実装（urdf_parser_pyが使えない場合）"""
    # 既存のXMLパーサーロジックをここに実装（省略）
    # 簡略化のため、空のspecsを返す
    return {
        "total_weight": 0.0,
        "base_length": 0.0,
        "base_width": 0.0,
        "base_height": 0.0,
        "thigh_length": 0.0,
        "calf_length": 0.0,
        "height_fully_standing": 0.0,
    }

def parse_urdf_specs(urdf_path, base_link_name, thigh_joint_patterns=None, calf_joint_patterns=None, known_values=None, mesh_measurements=None):
    """URDFファイルを解析してロボットの仕様を抽出（URDFから取得できる情報のみ、Mesh計測値も使用可能）"""
    if not os.path.exists(urdf_path):
        print(f"警告: URDFファイルが見つかりません: {urdf_path}")
        return None
    
    if USE_URDF_PARSER:
        return parse_urdf_specs_urdf_parser(urdf_path, base_link_name, mesh_measurements)
    else:
        return parse_urdf_specs_xml(urdf_path, base_link_name, thigh_joint_patterns, calf_joint_patterns, known_values)

ROBOT_ORDER = ["Go1", "Go2", "A1", "Mini Cheetah", "ANYmalC", "SpotMicro"]


def create_table(robot_specs):
    """ロボット仕様の表をMarkdown形式で作成"""
    header = "| Parameter | " + " | ".join(ROBOT_ORDER) + " |"
    sep = "|-----------|" + "|".join(["--------" for _ in ROBOT_ORDER]) + "|"
    md_table = """# Robot Specifications Table

URDFファイルから抽出した各ロボットの物理パラメータ（長さ・質量）．

""" + header + "\n" + sep + "\n"
    
    # パラメータの順序
    params = [
        ("Total weight (kg)", "total_weight"),
        ("Base length (m)", "base_length"),
        ("Base width (m)", "base_width"),
        ("Base height (m)", "base_height"),
        ("Thigh Length (m)", "thigh_length"),
        ("Calf Length (m)", "calf_length"),
    ]
    
    for param_name, param_key in params:
        row = f"| {param_name} |"
        for robot_name in ROBOT_ORDER:
            if robot_name in robot_specs and robot_specs[robot_name] is not None:
                value = robot_specs[robot_name].get(param_key, 0.0)
                # 値が0または非常に小さい場合は、取得できていないと判断して"-"を表示
                if value < 0.001 and param_key != "total_weight":
                    row += " - |"
                elif param_key == "total_weight":
                    row += f" {value:.1f} |"
                else:
                    row += f" {value:.2f} |"
            else:
                row += " - |"
        md_table += row + "\n"
    
    return md_table

# メイン処理
robot_specs = {}

for robot_name, config in ROBOT_CONFIGS.items():
    print(f"\n処理中: {robot_name}")
    print(f"URDFパス: {config['urdf_path']}")
    
    specs = parse_urdf_specs(
        config['urdf_path'], 
        config['base_link_name'],
        mesh_measurements=config.get('mesh_measurements', None)
    )
    
    if specs:
        robot_specs[robot_name] = specs
        print(f"  総重量: {specs['total_weight']:.2f} kg")
        print(f"  ベースサイズ: {specs['base_length']:.3f} x {specs['base_width']:.3f} x {specs['base_height']:.3f} m")
        print(f"  太もも長さ: {specs['thigh_length']:.3f} m")
        print(f"  すね長さ: {specs['calf_length']:.3f} m")
    else:
        robot_specs[robot_name] = None
        print(f"  エラー: 仕様の抽出に失敗しました")

# 表を作成
md_table = create_table(robot_specs)

# ファイルに保存
output_path = str(_SCRIPT_DIR / "robot_specifications_table.md")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_table)

print(f"\n{'='*60}")
print(f"表を保存しました: {output_path}")
print(f"{'='*60}")
print("\n作成された表:")
print(md_table)
