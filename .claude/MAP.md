# ANIMA Shared Infrastructure Map
# Updated: 2026-04-04
# Location: /mnt/forge-data/shared_infra/
# Repo: github.com/RobotFlow-Labs/anima-cuda-infra

## Quick Start
```bash
bash /mnt/forge-data/shared_infra/bootstrap_module.sh /path/to/your/module
```

---

## CUDA Kernels (43 compiled, py3.11 + CUDA 12.8 + L4 sm_89)

### Core Infrastructure (Waves 1-7)

| # | Kernel | Speedup | Path | Used By |
|---|--------|---------|------|---------|
| 1 | Gaussian rasterizer | built-in | cuda_extensions/gaussian_semantic_rasterization/ | 3 SLAM, 3DGS |
| 2 | Deformable attention | built-in | cuda_extensions/deformable_attention/ | LOKI, DETR modules |
| 3 | EAA renderer | JIT | cuda_extensions/eaa_renderer/ | MipSLAM, anti-aliased SLAM |
| 4 | Batched 3D→2D projection | 18B pts/s | cuda_extensions/ | Calibration, LiDAR fusion |
| 5 | Trilinear voxelizer | **304x** | cuda_extensions/ | OccAny, Ghost-FWL, 3D occupancy |
| 6 | Batch voxelizer | 11x | cuda_extensions/ | 3D detection, occupancy |
| 7 | Fused scatter-aggregate | fused | cuda_extensions/ | Feature aggregation |
| 8 | Batched 3D IoU/GIoU/NMS | **15x** | cuda_extensions/ | 3D detection, UAV |
| 9 | Depth projection + Z-buffer | 5.4x | cuda_extensions/ | LiDAR→image, depth modules |
| 10 | Fused grid warp + sample | **43.5x** | cuda_extensions/ | Depth warping, SLAM |
| 11 | Sparse trilinear upsample | 6.4x | cuda_extensions/ | Sparse 3D grids |
| 12 | Farthest point sampling | 7.2x | cuda_extensions/ | Point cloud sampling |
| 13 | Vectorized NMS (2D+3D) | 3D only | cuda_extensions/ | 3D NMS (no torchvision equivalent) |
| 14 | Sparse 3D convolution | hash table | cuda_extensions/ | Sparse 3D detection |
| 15 | SE(3) transform | **30x** | cuda_extensions/ | Point cloud transforms |
| 16 | Vector quantization | shared mem | cuda_extensions/ | Tokenization |
| 17 | **Seg argmax + colorize** | **2.6x** | cuda_extensions/seg_argmax_colorize/ | VIS-FORESTSIM, segmentation modules |
| 18 | **Terrain roughness** | **<0.1ms** | cuda_extensions/terrain_roughness/ | VIS-FORESTSIM, off-road navigation |
| 19 | **Mask morphology** | **5ms** | cuda_extensions/mask_morphology/ | VIS-FORESTSIM, mask cleanup |

### Wave 8 Defense Kernels (18 new)

| # | Kernel | Ops | Path | Used By |
|---|--------|-----|------|---------|
| 20 | UV texture remap + relight + composite | 3 | cuda_extensions/advreal_texture_remap/ | DEF-advreal (physical adversarial patches) |
| 21 | Smooth clamp + local TV + dual normalize | 3 | cuda_extensions/attackvla_patch_guard/ | DEF-attackvla (VLA defense) |
| 22 | Differentiable voxelization + BEV scatter | 2 | cuda_extensions/bevrobust_diff_voxelize/ | DEF-bevrobust (BEV 3D robustness) |
| 23 | Cross-modal interleave + gate norm | 3 | cuda_extensions/cmssm_interleave/ | DEF-cmssm (Mamba RGB-T fusion) |
| 24 | Chamfer distance + ball query group | 2 | cuda_extensions/desenat_pointcloud/ | DEF-desenat (point cloud defense) |
| 25 | DHiF unfold L2norm + IR enhance + LCN | 3 | cuda_extensions/dhif_ir_enhance/ | DEF-dhif (IR small target detection) |
| 26 | Modality select + saliency to box | 2 | cuda_extensions/hypsam_rgbt_saliency/ | DEF-hypsam (RGB-T SAM saliency) |
| 27 | Tiled Chamfer + 3D IoU + grad norms | 3 | cuda_extensions/iousattack_3d_ops/ | DEF-iousattack (LiDAR 3D attacks) |
| 28 | Parallel pixel sample + batch reward | 2 | cuda_extensions/rfpar_pixel_attack/ | DEF-rfpar (RL pixel attack) |
| 29 | Spatial distance decay + density blend | 2 | cuda_extensions/rgbtcc_crowd_count/ | DEF-rgbtcc (RGB-T crowd counting) |
| 30 | Patch apply + action perturb | 2 | cuda_extensions/roboticattack_vla/ | DEF-roboticattack (VLA robotic attack) |
| 31 | Gated fusion + channel L2 norm | 2 | cuda_extensions/rtfdnet_rgbt/ | DEF-rtfdnet (RGB-T fusion detection) |
| 32 | SAR patch speckle + magnitude | 2 | cuda_extensions/saap_sar/ | DEF-saap (SAR adversarial patch) |
| 33 | SAR log normalize + anomaly score | 2 | cuda_extensions/sariad_sar_anomaly/ | DEF-sariad (SAR anomaly detection) |
| 34 | Local entropy + patch inpaint | 2 | cuda_extensions/spann_patch_defense/ | DEF-spann (patch defense) |
| 35 | RGB-HSV perturb | 1 | cuda_extensions/tld_traffic_light/ | DEF-tld (traffic light attacks) |
| 36 | RGB-T concat norm + seg argmax | 2 | cuda_extensions/tuni_rgbt_seg/ | DEF-tuni (RGB-T segmentation) |
| 37 | DETR bias GELU + box decode | 2 | cuda_extensions/uavdetr_detr/ | DEF-uavdetr (counter-UAV DETR) |

### Domain-Group Shared Kernels (6 new, covers all 73 project_* modules)

| # | Kernel Package | Ops | Path | Used By |
|---|---------------|-----|------|---------|
| 38 | Fused image preprocess | 4 | cuda_extensions/fused_image_preprocess/ | ALL 73 modules (normalize, augment, resize) |
| 39 | Gaussian splatting ops | 4 | cuda_extensions/gaussian_ops/ | 8 SLAM/3DGS modules (huginn, freya, mimir, etc.) |
| 40 | Detection ops | 4 | cuda_extensions/detection_ops/ | 11 detection modules (azoth, loki, panoptes, etc.) |
| 41 | Depth estimation ops | 4 | cuda_extensions/depth_estimation_ops/ | 8 depth modules (abyssos, frigg, grid, etc.) |
| 42 | Robotics / VLA ops | 4 | cuda_extensions/robotics_ops/ | 11 robotics modules (centaur, fenrir, morpheus, etc.) |
| 43 | Navigation ops | 3 | cuda_extensions/navigation_ops/ | 6 navigation modules (geri, hermes, njord, etc.) |
| 44 | Scene fusion ops | 2 | cuda_extensions/scene_fusion_ops/ | 10 scene/occupancy modules (logos, urd, eir, etc.) |

All kernels have PyTorch CPU fallback. Install: `uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl`

---

## Pre-Computed Dataset Caches (565GB total)

| # | Cache | Size | Path | Used By |
|---|-------|------|------|---------|
| 1 | nuScenes voxels (Occ3D) | 163GB | shared_infra/datasets/nuscenes_voxels/ | OccAny, SURT, DTP, occupancy modules |
| 2 | nuScenes DINOv2-B/14 | 140GB | shared_infra/datasets/nuscenes_dinov2_features/ | OccAny, ProjFusion, DTP |
| 3 | KITTI voxels | 117GB | shared_infra/datasets/kitti_voxel_cache/ | Ghost-FWL, ProjFusion, OccAny |
| 4 | SERAPHIM VLM features | 53GB | shared_infra/datasets/seraphim_vlm_features/ | UAVDETR, TrackVLA, UAV modules |
| 5 | COCO HDINO | 25GB | shared_infra/datasets/coco_hdino_cache/ | LOKI, DETR modules |
| 6 | SERAPHIM tensors | 23GB | shared_infra/datasets/seraphim_tensor_cache/ | UAVDETR, TrackVLA |
| 7 | TUM DINOv2 | 12GB | shared_infra/datasets/tum_dinov2_features/ | SLAM modules |
| 8 | COCO DINOv2 | 9.9GB | shared_infra/datasets/coco_dinov2_features/ | Detection modules |
| 9 | COCO SAM2 | 9.8GB | shared_infra/datasets/coco_sam2_features/ | Segmentation, RPGA |
| 10 | KITTI depth | 6.6GB | shared_infra/datasets/kitti_depth_cache/ | ProjFusion, Ghost-FWL |
| 11 | Replica DINOv2 | 3.0GB | shared_infra/datasets/replica_dinov2_features/ | SLAM modules |
| 12 | KITTI DINOv2 | 2.8GB | shared_infra/datasets/kitti_dinov2_features/ | ProjFusion, OccAny |
| 13 | nuScenes pointclouds | 1.9GB | shared_infra/datasets/nuscenes_pointcloud_cache/ | ProjFusion, DTP |
| 14 | KITTI pointclouds | 381MB | shared_infra/datasets/kitti_pointcloud_cache/ | Ghost-FWL, ProjFusion |

**DO NOT re-compute these. Load directly from cache paths.**

---

## Raw Datasets (already on disk — DO NOT download)

| Dataset | Path |
|---------|------|
| COCO val+train | /mnt/forge-data/datasets/coco/ + /mnt/train-data/datasets/coco/ |
| nuScenes | /mnt/forge-data/datasets/nuscenes/ |
| KITTI | /mnt/forge-data/datasets/kitti/ |
| TUM RGB-D / TUM-VI | /mnt/forge-data/datasets/tum/ |
| Replica raw meshes | /mnt/forge-data/datasets/replica/ |
| Replica RGB-D rendered | /mnt/forge-data/datasets/replica_rgbd/ (17GB, 6 scenes) |
| Replica SLAM 2-agent | /mnt/forge-data/datasets/replica_slam/ (895MB) |
| COD10K | /mnt/forge-data/datasets/cod10k/ |
| MCOD | /mnt/forge-data/datasets/mcod/ |
| NUAA-SIRST | /mnt/forge-data/datasets/nuaa_sirst_yolo/ |
| SERAPHIM UAV (83K) | /mnt/forge-data/datasets/uav_detection/seraphim/ (8.6GB) |
| ForestSim raw (2093 pairs) | /mnt/forge-data/datasets/vision/forestsim/raw/ (8GB) |
| ForestSim 24-class MMSeg | /mnt/forge-data/datasets/vision/forestsim/forestsim_all/ |
| ForestSim 6-class traversability | /mnt/forge-data/datasets/vision/forestsim/forestsim_group6/ |

## Models (already on disk — DO NOT download)

| Model | Path |
|-------|------|
| DINOv2 ViT-B/14 | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth |
| DINOv2 ViT-G/14 | /mnt/forge-data/models/dinov2_vitg14_reg4_pretrain.pth |
| DINOv2-Small | /mnt/forge-data/models/facebook--dinov2-small/ |
| SAM ViT-B | /mnt/forge-data/models/sam_vit_b_01ec64.pth |
| SAM ViT-H | /mnt/forge-data/models/sam_vit_h_4b8939.pth |
| SAM 2.1 base | /mnt/forge-data/models/sam2.1_hiera_base_plus.pt |
| SAM 2.1 large | /mnt/forge-data/models/sam2.1-hiera-large/ |
| GroundingDINO | /mnt/forge-data/models/groundingdino_swint_ogc.pth |
| YOLOv5l6 | /mnt/forge-data/models/yolov5l6.pt |
| YOLOv12n | /mnt/forge-data/models/yolov12n.pt |
| YOLO11n | /mnt/forge-data/models/yolo11n.pt |
| OccAny checkpoints | /mnt/forge-data/models/occany/ (23GB) |
| HaMeR | /mnt/forge-data/models/hamer_demo_data/ (6GB) |
| Handy hand model | /mnt/forge-data/models/handy/ |
| NIMBLE hand model | /mnt/forge-data/models/nimble/ |
| SigLIP-2 | /mnt/forge-data/models/siglip2-base-patch16-384/ |
| CLIP ViT-B/32 | /mnt/forge-data/models/clip-vit-base-patch32/ |
| Stable Diffusion 2.1 | /mnt/forge-data/models/stable-diffusion-2-1/ |
| OVIE generator | /mnt/forge-data/models/kyutai-ovie/ |
| OVIE eval | /mnt/forge-data/models/ovie/ovie.pt |

## Tools

| Tool | Path | Usage |
|------|------|-------|
| TRT export toolkit | shared_infra/trt_toolkit/export_to_trt.py | ONNX → TRT fp16 + fp32 |
| Bootstrap script | shared_infra/bootstrap_module.sh | New module → ready in 5 min |
| CUDA 12 build script | shared_infra/cuda_extensions/BUILD_CU128.sh | Force cu128 compilation |
| anima-cuda-infra repo | github.com/RobotFlow-Labs/anima-cuda-infra | All 44 kernel packages + benchmarks |

## Output Paths

| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/{module_name}/ |
| Logs | /mnt/artifacts-datai/logs/{module_name}/ |
| Exports | /mnt/artifacts-datai/exports/{module_name}/ |

## Rules
- ALWAYS install torch with cu128: --index-url https://download.pytorch.org/whl/cu128
- NEVER download datasets/models that already exist on disk
- NEVER re-compute cached features — load from shared_infra/datasets/
- ALWAYS use shared CUDA kernels from shared_infra/cuda_extensions/
- ALWAYS use nohup+disown for training
- ALWAYS export TRT FP16 + TRT FP32 (use shared toolkit)
- Save ANY new CUDA kernels to shared_infra/cuda_extensions/
- Save ANY new pre-processed data to shared_infra/datasets/
- Run /anima-optimize-cuda-pipeline to profile + optimize before training

## KNOWN LARGE DATASETS (not downloaded, available if needed)

| Dataset | Size | Source | Good For |
|---------|------|--------|----------|
| The Well | 15TB | https://github.com/PolymathicAI/the_well/ | Physics simulation, PDE solving, scientific ML |
| ImageNet Full | 1.2TB | image-net.org (registration) | Classification, adversarial |
| Waymo Open | ~1TB | waymo.com (gated) | Autonomous driving |
