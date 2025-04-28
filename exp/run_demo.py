from lib.kits.hsmr_demo import *
import cv2
import numpy as np
from pathlib import Path
import torch
import os

# 启用 PyTorch 内存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.
    with monitor('Data Preprocess'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args)

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                batch_size=args.det_bs,
                max_img_size=args.det_mis,
                device=args.device,
            )
            torch.cuda.empty_cache()  # 释放显存

    # ⛩️ 2. Process inputs.
    if inputs_meta['type'] in ['camera', 'stream']:
        # 实时处理：逐帧处理
        cv2.namedWindow('HSMR Demo', cv2.WINDOW_NORMAL)
        frame_idx = 0
        save_video_flag = True
        video_writer = None
        frame_width, frame_height = None, None

        # 初始化推理管道
        with monitor('Pipeline Initialization'):
            get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
            pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)
            torch.cuda.empty_cache()  # 释放显存

        for frame in raw_imgs:
            with monitor('Detecting'):
                frame_list = [frame]
                detector_outputs = detector(frame_list)
                patches, det_meta = imgs_det2patches(
                    frame_list, *detector_outputs, args.max_instances
                )
                torch.cuda.empty_cache()  # 释放显存

            with monitor('Recovery'):
                if len(patches) == 0:
                    get_logger(brief=True).warning(f'No human instance detected in frame {frame_idx}.')
                    result_frame = frame
                else:
                    get_logger(brief=True).info(f'🔍 {len(patches)} human instances detected in frame {frame_idx}.')
                    patches_i = np.concatenate(patches, axis=0)
                    patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255
                    patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)
                    with torch.no_grad():
                        outputs = pipeline(patches_normalized_i)
                    pd_params = {k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()}
                    pd_cam_t = outputs['pd_cam_t'].detach().cpu().clone()

                    get_logger(brief=True).info(f'🤌 Preparing meshes...')
                    m_skin, m_skel = prepare_mesh(pipeline, pd_params)

                    with monitor('Visualization'):
                        m_skel = None if args.ignore_skel else m_skel
                        results, full_cam_t = visualize_full_img(
                            pd_cam_t, frame_list, det_meta, m_skin, m_skel, args.have_caption
                        )
                        result_frame = results[0]

            # 显示结果
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('HSMR Demo', result_frame_bgr)

            # 保存视频
            if save_video_flag:
                if video_writer is None:
                    frame_height, frame_width = result_frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_path = outputs_root / f'{pipeline.name}-{inputs_meta["seq_name"]}.mp4'
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (frame_width, frame_height))
                video_writer.write(result_frame_bgr)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        # 清理
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    else:
        # 静态图像或视频处理（原始逻辑）
        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)
            torch.cuda.empty_cache()  # 释放显存

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)
            if len(patches) == 0:
                get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
                return
            get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')

        # 初始化推理管道（延迟到检测后）
        with monitor('Pipeline Initialization'):
            get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
            pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)
            torch.cuda.empty_cache()  # 释放显存

        with monitor('Recovery'):
            get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
            pd_params, pd_cam_t = [], []
            for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
                patches_i = np.concatenate(patches[bw.sid:bw.eid], axis=0)
                patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255
                patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)
                with torch.no_grad():
                    outputs = pipeline(patches_normalized_i)
                pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
                pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

            pd_params = assemble_dict(pd_params, expand_dim=False)
            pd_cam_t = torch.cat(pd_cam_t, dim=0)
            dump_results = {
                'patch_cam_t': pd_cam_t.numpy(),
                **{k: v.numpy() for k, v in pd_params.items()},
            }

            get_logger(brief=True).info(f'🤌 Preparing meshes...')
            m_skin, m_skel = prepare_mesh(pipeline, pd_params)
            get_logger(brief=True).info(f'🏁 Done.')

        with monitor('Visualization'):
            if args.ignore_skel:
                m_skel = None
            results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption)
            dump_results['full_cam_t'] = full_cam_t
            # Save rendering and dump results.
            if inputs_meta['type'] == 'video':
                seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
                save_video(results, outputs_root / f'{seq_name}.mp4')
                np.savez(outputs_root / f'{seq_name}.npz', **dump_results)
            elif inputs_meta['type'] == 'imgs':
                img_names = [f'{pipeline.name}-{fn.name}' for fn in inputs_meta['img_fns']]
                dump_results = disassemble_dict(dump_results, keep_dim=True)
                for i, img_name in enumerate(tqdm(img_names, desc='Saving images')):
                    save_img(results[i], outputs_root / f'{img_name}.jpg')
                    np.savez(outputs_root / f'{img_name}.npz', **dump_results[i])

            get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')

    get_logger(brief=True).info(f'🎊 Everything is done!')
    monitor.report()


if __name__ == '__main__':
    main()