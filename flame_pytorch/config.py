from dataclasses import dataclass

@dataclass
class FLAMEConfig:
    flame_model_path: str = "./model/generic_model.pkl"
    static_landmark_embedding_path: str = "./model/flame_static_embedding.pkl"
    dynamic_landmark_embedding_path: str = "./model/flame_dynamic_embedding.npy"
    shape_params: int = 100
    expression_params: int = 50
    pose_params: int = 6
    use_face_contour: bool = True
    use_3D_translation: bool = True
    optimize_eyeballpose: bool = True
    optimize_neckpose: bool = True
    num_worker: int = 4
    batch_size: int = 8
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0
