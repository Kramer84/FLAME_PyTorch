import numpy as np
import pyrender
import torch
import trimesh
import typer

from flame_pytorch import FLAME, FLAMEConfig

app = typer.Typer()

@app.command()
def main(
    flame_model_path: str = "./model/generic_model.pkl",
    static_landmark_embedding_path: str = "./model/flame_static_embedding.pkl",
    dynamic_landmark_embedding_path: str = "./model/flame_dynamic_embedding.npy",
    shape_params: int = 100,
    expression_params: int = 50,
    pose_params: int = 6,
    use_face_contour: bool = True,
    use_3D_translation: bool = True,
    optimize_eyeballpose: bool = True,
    optimize_neckpose: bool = True,
    batch_size: int = 8,
):
    # Map CLI arguments to the clean configuration object
    config = FLAMEConfig(
        flame_model_path=flame_model_path,
        static_landmark_embedding_path=static_landmark_embedding_path,
        dynamic_landmark_embedding_path=dynamic_landmark_embedding_path,
        shape_params=shape_params,
        expression_params=expression_params,
        pose_params=pose_params,
        use_face_contour=use_face_contour,
        use_3D_translation=use_3D_translation,
        optimize_eyeballpose=optimize_eyeballpose,
        optimize_neckpose=optimize_neckpose,
        batch_size=batch_size,
    )

    radian = np.pi / 180.0
    flamelayer = FLAME(config)

    shape_params_tensor = torch.zeros(batch_size, shape_params).cuda()

    # Note: Hardcoded to 8 items in the original code.
    # If batch_size differs from 8 via CLI, this array logic will need resizing.
    pose_params_numpy = np.array(
        [
            [0.0, 30.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, -30.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, 85.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, -48.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, -15.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0 * radian, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.0 * radian, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    if batch_size != 8:
         pose_params_numpy = np.resize(pose_params_numpy, (batch_size, 6))

    pose_params_tensor = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
    expression_params_tensor = torch.zeros(batch_size, expression_params, dtype=torch.float32).cuda()

    flamelayer.cuda()
    vertice, landmark = flamelayer(shape_params_tensor, expression_params_tensor, pose_params_tensor)
    print(vertice.size(), landmark.size())

    if config.optimize_eyeballpose and config.optimize_neckpose:
        neck_pose = torch.zeros(batch_size, 3).cuda()
        eye_pose = torch.zeros(batch_size, 6).cuda()
        vertice, landmark = flamelayer(
            shape_params_tensor, expression_params_tensor, pose_params_tensor, neck_pose, eye_pose
        )

    faces = flamelayer.faces
    for i in range(batch_size):
        vertices = vertice[i].detach().cpu().numpy().squeeze()
        joints = landmark[i].detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    app()
