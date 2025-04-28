from lib.kits.hsmr_demo import *
import cv2
import torch
import open3d as o3d
import numpy as np

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    monitor = TimeMonitor()

    # ⛩️ 1. Initialize Detector and Pipeline.
    with monitor('Detector Initialization'):
        get_logger(brief=True).info('🧱 Building detector.')
        detector = build_detector(
            batch_size=args.det_bs,
            max_img_size=args.det_mis,
            device=args.device,
        )

    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info('🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    # ⛩️ 2. Setup Camera and Visualization.
    cap = cv2.VideoCapture(0)  # 打开摄像头
    if not cap.isOpened():
        get_logger(brief=True).error('🚫 Failed to open camera!')
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='HSMR Real-Time Demo')

    # ⛩️ 3. Real-Time Loop.**************************
    get_logger(brief=True).info('🎥 Starting real-time demo...')
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            get_logger(brief=True).error('🚫 Failed to read frame from camera!')
            break

        # 预处理：检测人体实例
        with monitor('Detecting'):
            raw_imgs = [frame]  # 单帧输入
            detector_outputs = detector(raw_imgs)
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)
            if len(patches) == 0:
                get_logger(brief=True).warning('🚫 No human instance detected in this frame.')
                continue
            get_logger(brief=True).info(f'🔍 Detected {len(patches)} human instances.')

        # 推理：逐帧恢复骨骼和网格
        with monitor('Recovery'):
            patches_i = np.concatenate(patches, axis=0)  # (N, 256, 256, 3)
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            pd_params = {k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()}
            pd_cam_t = outputs['pd_cam_t'].detach().cpu().clone()

            # 准备网格
            m_skin, m_skel = prepare_mesh(pipeline, [pd_params])  # 单帧输入，列表长度为1

        # 可视化：实时渲染
        with monitor('Visualization'):
            if args.ignore_skel:
                m_skel = None
            results, full_cam_t = visualize_full_img(
                pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption
            )

            # 使用Open3D渲染（假设m_skin和m_skel是Open3D几何对象）
            vis.clear_geometries()
            if m_skin:
                vis.add_geometry(m_skin[0])  # 单帧结果
            if m_skel:
                vis.add_geometry(m_skel[0])
            vis.poll_events()
            vis.update_renderer()

        # 显示原始图像（可选）
        cv2.imshow('Camera Input', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ⛩️ 4. Cleanup.
    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
    get_logger(brief=True).info('🎊 Real-time demo finished!')
    monitor.report()

if __name__ == '__main__':
    main()