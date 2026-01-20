import os
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d
import matplotlib
matplotlib.use('Agg') 
import torch
from utils.dataset import TMPPDataset
from prompts.get_prompts import get_prompts
from openai import OpenAI
import io
import base64
import json
import re
import tasmap_pp_visualizer as viz
import shutil
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


task_dict = {
    'relocate': 0,
    'reorient': 1,
    'washing-up': 2,
    'mop': 3,
    'vacuum': 4,
    'wipe': 5,
    'fold': 6,
    'close': 7,
    'turn-off': 8,
    'dispose': 9,
    'empty': 10,
    'leave': 11
}



class TMPPInference:
    def __init__(self, root_env_dir, seq_name, seg3D_path, save_root_path, model_name, api_key, debug=False, device='cuda'):
        self.seq_name = seq_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.datasets = TMPPDataset(root_env_dir=root_env_dir)
        self.dataset = self.datasets(seq_name)

        self.save_root_path = save_root_path
        self.point_cloud_path = self.dataset.point_cloud_path
        self.debug = debug
        self.max_retries = 3    # retry gpt maximum 3 times
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

        self.object_dict_path = os.path.join(seg3D_path, 'object_dict.npy')
        self.mask_path = os.path.join(seg3D_path, 'class_agnostic_mask.npz')
        self.save_gpt_path = os.path.join(save_root_path, f'{model_name}')
        self.debug_root_path = os.path.join(self.save_gpt_path, 'debug_images')
        self.if_result_dir = os.path.join(self.save_gpt_path, 'if_result')

        self.if_results = {}
        self.if_results_file = os.path.join(self.if_result_dir, 'results.json')

        self.debug_bbox_dir = os.path.join(self.debug_root_path, 'bbox')
        self.debug_if_result_images_dir = os.path.join(self.debug_root_path, 'if_result_images')

    def _process_single_object(self, key, value, masks, scene_points, messages):
        try:
            mask_list = value['mask_list'][:20]
            if len(mask_list) == 0:
                return None

            mask = masks[:, key]
            point_ids = np.where(mask)[0]
            points = scene_points[point_ids]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            aabb = pcd.get_axis_aligned_bounding_box()
            size = aabb.get_extent()
            if np.any(size > 4) or size[2] > 2.3:
                return None

            response_text_path = os.path.join(self.if_result_dir, f'{key}_response.txt')
            
            grid_image = self.__get_grid_image(mask_list, key, pcd)
            
            if not grid_image:
                return None

            if os.path.exists(response_text_path):
                with open(response_text_path, 'r') as rf:
                    response = rf.read()
            else:
                response = self.__inference(messages, grid_image)
                with open(response_text_path, 'w') as rf:
                    rf.write(response)

            target, task = self.__parse_respose(response)
            
            if self.debug:
                grid_image.save(os.path.join(self.debug_if_result_images_dir, f"{key}_{target}_{task}.png"))

            return key, {"target": target, "task": task}, response
            
        except Exception as e:
            print(f"Error processing object {key}: {e}")
            return None


    def run(self, max_workers=None):
        # if os.path.exists(self.save_gpt_path):
        #     shutil.rmtree(self.save_gpt_path)
        os.makedirs(self.if_result_dir, exist_ok=True)

        if os.path.exists(self.debug_root_path):
            shutil.rmtree(self.debug_root_path)

        if self.debug:
            os.makedirs(self.debug_root_path, exist_ok=True)
            os.makedirs(self.debug_bbox_dir, exist_ok=True)
            os.makedirs(self.debug_if_result_images_dir, exist_ok=True)

        object_dict = np.load(self.object_dict_path, allow_pickle=True).item()
        pred = np.load(self.mask_path)
        masks = pred['pred_masks']
        mesh = o3d.io.read_triangle_mesh(self.point_cloud_path)
        scene_points = np.asarray(mesh.vertices)
        messages, _ = get_prompts()
        if max_workers is None:

            pbar = tqdm(total=len(object_dict), dynamic_ncols=True, desc=f"[{self.seq_name}] Inferencing")
            for key, value in object_dict.items():
                mask_list = value['mask_list'][:20]

                if len(mask_list)==0: continue
                response_text_path = os.path.join(self.if_result_dir, f'{key}_response.txt')
                # if os.path.exists(response_text_path):
                #     continue

                mask = masks[:, key]
                point_ids = np.where(mask)[0]
                points = scene_points[point_ids]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # floor, wall 제외
                aabb = pcd.get_axis_aligned_bounding_box()
                size = aabb.get_extent()
                if np.any(size>4) or size[2]>2.3: 
                    continue

                grid_image = self.__get_grid_image(mask_list, key, pcd)
                if grid_image: 
                    if os.path.exists(response_text_path):
                        with open(response_text_path, 'r') as rf:
                            response = rf.read()
                    else:
                        response = self.__inference(messages, grid_image)
                        with open(response_text_path, 'w') as rf:
                            rf.write(response)
                        # response = 'None'
                    tqdm.write(f"\n=== Object {key} ===\n{response}")

                    target, task = self.__parse_respose(response)
                    self.if_results[key] = {"target":target,
                                            "task":task}
                    if self.debug:
                        grid_image.save(os.path.join(self.debug_if_result_images_dir, f"{key}_{target}_{task}.png"))
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for key, value in object_dict.items():
                    futures.append(
                        executor.submit(
                            self._process_single_object, 
                            key, value, masks, scene_points, messages
                        )
                    )

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{self.seq_name}] Inferencing"):
                    result = future.result()
                    if result:
                        key, res_dict, response_text = result
                        self.if_results[key] = res_dict
                        
                        tqdm.write(f"\n=== Object {key} ===\n{response_text}")
        
        with open(self.if_results_file, "w", encoding="utf-8") as jf:
            json.dump(self.if_results, jf, indent=4, ensure_ascii=False)

        
    def semantic_fusion(self):
        result_save_path = os.path.join(self.save_root_path, "tasmap_pp.npy")
        result_data = []

        object_dict = np.load(self.object_dict_path, allow_pickle=True).item()
        pred = np.load(self.mask_path)
        masks = pred['pred_masks']
        pcd = o3d.t.io.read_point_cloud(self.point_cloud_path)


        # segment_GT_id = pcd.point["segment_GT_id"].numpy()
        pcd_legacy = pcd.to_legacy()
        scene_points = np.asarray(pcd_legacy.points).astype(np.float32)
        scene_colors = np.asarray(pcd_legacy.colors).astype(np.float32)

        pbar = tqdm(total=len(object_dict), dynamic_ncols=True, desc=f"[{self.seq_name}] Saving")
        for key, value in object_dict.items():
            mask_list = value['mask_list']
            # if len(mask_list) == 0:
            #     continue

            response_text_path = os.path.join(self.if_result_dir, f'{key}_response.txt')
            if os.path.exists(response_text_path):
                with open(response_text_path, 'r') as rf:
                    response = rf.read()
                target, task = self.__parse_respose(response)
            else:
                target = None
                task = None

            mask = masks[:, key]  # shape: (num_points,)
            point_ids = np.where(mask)[0]

            for pid in point_ids:
                point_info = {
                    'key': key,
                    'position': scene_points[pid],
                    'color': scene_colors[pid],
                    # 'segment_GT_id': segment_GT_id[pid],
                    'target': target,
                    'task': task
                }
                result_data.append(point_info)

            pbar.update(1)

        np.save(result_save_path, result_data)


    def visualize(self):
        if os.path.isfile(self.if_results_file):
            with open(self.if_results_file, 'r') as jf:
                self.if_results = json.load(jf)
        np.random.seed(35)

        def compute_obb(points: np.ndarray):
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            center = ((min_pt + max_pt) / 2)
            padding = 0.02
            size = (max_pt - min_pt) + padding
            z_diff = center[2]-size[2]/2-padding/2

            if z_diff < 0:
                center += [0,0,-z_diff]
                size -= [0,0,-z_diff/2]

            rotation = np.array([0,0,0,1])
            return center, size, rotation

        
        def vis_one_object(point_ids, scene_points):
            points = scene_points[point_ids]
            color = (np.random.rand(3) * 0.7 + 0.3) * 255
            colors = np.tile(color, (points.shape[0], 1))
            return point_ids, points, colors, color, np.mean(points, axis=0)
        
        point_size = 10 #20

        pcd = o3d.t.io.read_point_cloud(self.dataset.point_cloud_path)
        scene_points = pcd.point["positions"].numpy()
        scene_colors = pcd.point["colors"].numpy()

        # Since the color of raw scan may be too dark, we brighten it tone mapping
        scene_colors = np.power(scene_colors, 1/1.4)
        scene_colors = scene_colors * 255

        instance_colors = np.zeros_like(scene_colors)

        v = viz.TASMapVisualizer()

        pred = np.load(self.mask_path)
        masks = pred['pred_masks']

        num_instances = masks.shape[1]
        for idx in range(num_instances):
            mask = masks[:, idx]
            point_ids = np.where(mask)[0]

            point_ids, points, colors, label_color, center = vis_one_object(point_ids, scene_points)
            instance_colors[point_ids] = label_color
            if str(idx) in self.if_results.keys():
                position, size, rotation = compute_obb(points)

                v.add_cuboid(    
                    name=f'bboxes;{idx}',
                    label=f"{self.if_results[str(idx)]['target']}",
                    task=f"[{self.if_results[str(idx)]['task']}]",
                    position=position,
                    size=size,
                    rotation=rotation,
                    color=np.array([255, 0, 0]),  # red
                    alpha=1,
                    edge_width= 0.004 ,
                    visible=True,)

            v.add_points(f'instances;{idx}', points, colors, visible=False, point_size=point_size)


        vox = pcd.voxel_down_sample(voxel_size=0.01)
        positions = vox.point["positions"]
        z = positions[:, 2]

        mask = z <= 2.3
        filtered_vox = vox.select_by_mask(mask)

        RGB_points = filtered_vox.point["positions"].numpy()
        RGB_colors = filtered_vox.point["colors"].numpy()
        RGB_colors = np.power(RGB_colors, 1/1.4)
        RGB_colors = RGB_colors * 255
        v.add_points('RGB', RGB_points, RGB_colors, visible=True, point_size=point_size)


        vis_path = f'{self.save_root_path}/vis'
        if os.path.isdir(vis_path):
            shutil.rmtree(vis_path)
        v.save(vis_path)



    def __parse_respose(self, response):
        try:
            json_text_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_text_match:
                json_text = json_text_match.group(0)
                data = json.loads(json_text)
                target = data.get("target", "").split('/')[0]
                if type(data.get("task", "")) is list:
                    if type(data.get("task", "")[0]) is dict:
                        task_dict = data.get("task", "")
                        task = ",".join([f"{item['name'].split()[0]}({item['confidence']})" for item in task_dict])
                    else:
                        task_list = data.get("task", "")
                        task = ",".join([item.split()[0] for item in task_list])
                else:
                    task = data.get("task", "").split(" ")[0]
                return target, task
        except (json.JSONDecodeError, AttributeError):
            print("Parsing Error !!")
            pass
        
        return "", ""
    

    def __inference(self, messages, grid_image):
        buffered = io.BytesIO()
        messages = messages.copy()

        grid_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        query = [{
            "role": "user", 
            "content": [
                {"type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                },
                {"type": "text", 
                "text": "I will give you 100 bucks if you respond well. Analyze the image step by step."},
            ]
        }]
        messages.extend(query)

        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort="high",
                    response_format={"type": "text"
                    },
                    store=False
                )
                answer = response.choices[0].message.content
                target, task = self.__parse_respose(answer)
                if not target or not task:
                    raise ValueError("Parsing returned empty target/task")
                return answer
            except Exception as e:
                attempt += 1
                print(f"[Retry {attempt}/{self.max_retries}] Error: {e}")
                time.sleep(1) 
            return "resonse_error"

    def __mask2box(self, mask):
        pos = np.where(mask)
        top = np.min(pos[0])
        bottom = np.max(pos[0])
        left = np.min(pos[1])
        right = np.max(pos[1])
        return (left, top), (right, bottom)

    

    def __get_grid_image(self, mask_list, key, pcd):
        image_list = []
        valid_list = []
        bbox_idx = 0
        for mask_info in mask_list:
            if sum(valid_list) > 8: 
                continue
            frame_id, mask_id, conf = mask_info
            rgb_path, _ = self.dataset.get_frame_path(frame_id)
            intrinsic = self.dataset.get_intrinsics(frame_id)
            extrinsic = self.dataset.get_extrinsic(frame_id)
            segmentation_image = self.dataset.get_segmentation(frame_id)
            depth_map = self.dataset.get_depth(frame_id)

            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            mask = (segmentation_image == mask_id)

            try:
                bbox, valid, visible_ratio = self.__get_bbox_by_projection(pcd, intrinsic, extrinsic, rgb, mask, depth_map)
                try:
                    if visible_ratio<0.75:
                        continue
                except:
                    continue
            except:
                bbox, valid = None, False
            valid_list.append(valid)

            if valid:
                bbox_image = self.__draw_bbox(rgb, bbox, is_red=False)
                if self.debug:
                    debug_bbox_image = self.__draw_bbox(rgb, bbox)
                    filename = f"{key}_{conf:.3f}_{frame_id}_{bbox_idx}.jpg"
                    bbox_idx += 1
                    debug_bbox_image_bgr = cv2.cvtColor(debug_bbox_image, cv2.COLOR_RGB2BGR)
                    debug_bbox_image_bgr = cv2.resize(debug_bbox_image_bgr, (512, 512))
                    cv2.imwrite(os.path.join(self.debug_bbox_dir, filename), debug_bbox_image_bgr)
            else:
                bbox_image = None
            image_list.append(bbox_image)

        image_list = [img for img, valid in zip(image_list, valid_list) if valid]
        if len(image_list)<2:
            return None
        grid_image = self.__stitch_bbox_images(image_list)

        return grid_image


    def __project_pcd(self, pcd, intrinsic, extrinsics, depth_map):
        fx, fy = intrinsic.intrinsic_matrix[0, 0], intrinsic.intrinsic_matrix[1, 1]
        cx, cy = intrinsic.intrinsic_matrix[0, 2], intrinsic.intrinsic_matrix[1, 2]
        width, height = intrinsic.width, intrinsic.height

        world_to_cam = np.linalg.inv(extrinsics)

        def project_points(points):
            hom = np.hstack([points, np.ones((len(points), 1))])
            camera_points = (world_to_cam @ hom.T).T
            x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

            valid = z > 1e-6 
            x, y, z = x[valid], y[valid], z[valid]

            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            return u, v, z, camera_points[valid]

        # Project PCD
        points = np.asarray(pcd.points)
        u, v, z, cam_points = project_points(points)
        px = np.round(u).astype(int)
        py = np.round(v).astype(int)
        in_bounds = (0 <= px) & (px < width) & (0 <= py) & (py < height)

        if not np.any(in_bounds):
            return None, None

        # Occlusion test using depth map
        depth_tolerance = 0.2  # meter
        visible = []
        visible_px = []
        visible_py = []

        px_valid = px[in_bounds]
        py_valid = py[in_bounds]
        z_valid = z[in_bounds]

        for x_, y_, z_ in zip(px_valid, py_valid, z_valid):
            background_depth = depth_map[y_, x_]
            if np.isfinite(background_depth) and background_depth < z_ - depth_tolerance:
                continue
            visible.append(True)
            visible_px.append(x_)
            visible_py.append(y_)

        visible = np.array(visible)
        visible_ratio = len(visible) / len(points)
        in_bound_ratio = len(px_valid) / len(points)

        # Create mask only for visible pixels
        mask = np.zeros((height, width), dtype=bool)
        mask[visible_py, visible_px] = True

        return mask, visible_ratio



    def __get_bbox_by_projection(self, pcd, intrinsic, extrinsics, rgb, mask, depth_map):
        width, height = intrinsic.width, intrinsic.height

        pcd_mask, visible_ratio = self.__project_pcd(pcd, intrinsic, extrinsics, depth_map)

        total_bbox = self.__mask2box(pcd_mask)

        (xmin, ymin), (xmax, ymax) = total_bbox

        bbox_area = (xmax - xmin) * (ymax - ymin)
        bbox_ratio = bbox_area / (width * height)

        if bbox_ratio<0.9 and bbox_ratio>0.001:
            is_valid = True
        else:
            is_valid = False

        return (xmin, ymin, xmax, ymax), is_valid, visible_ratio


    def __crop_image(self, image, x_min, y_min, x_max, y_max, min_bbox_ratio):
        h, w = image.shape[:2]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_ratio = (bbox_width*bbox_height)/(h*w)

        scale_factor = min(math.sqrt(min_bbox_ratio / bbox_ratio), 4)

        crop_w = w / scale_factor
        crop_h = h / scale_factor

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        crop_x_min = center_x - crop_w / 2
        crop_y_min = center_y - crop_h / 2
        crop_x_max = center_x + crop_w / 2
        crop_y_max = center_y + crop_h / 2

        if crop_x_min < 0:
            crop_x_max -= crop_x_min 
            crop_x_min = 0
        if crop_y_min < 0:
            crop_y_max -= crop_y_min
            crop_y_min = 0
        if crop_x_max > w:
            diff = crop_x_max - w
            crop_x_min = max(0, crop_x_min - diff)
            crop_x_max = w
        if crop_y_max > h:
            diff = crop_y_max - h
            crop_y_min = max(0, crop_y_min - diff)
            crop_y_max = h

        crop_x_min_int = int(round(crop_x_min))
        crop_y_min_int = int(round(crop_y_min))
        crop_x_max_int = int(round(crop_x_max))
        crop_y_max_int = int(round(crop_y_max))

        cropped = image[crop_y_min_int:crop_y_max_int, crop_x_min_int:crop_x_max_int].copy()

        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        resize_scale = w / (crop_x_max_int - crop_x_min_int)

        new_x_min = int((x_min - crop_x_min_int) * resize_scale)
        new_y_min = int((y_min - crop_y_min_int) * resize_scale)
        new_x_max = int((x_max - crop_x_min_int) * resize_scale)
        new_y_max = int((y_max - crop_y_min_int) * resize_scale)

        return resized, new_x_min, new_y_min, new_x_max, new_y_max


    def __draw_bbox(self, image, bbox, thickness=2, padding=10, min_bbox_ratio = 0.05, is_red=True):
        result = image.copy()

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            margin = thickness // 2
            h, w = result.shape[:2]

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            padding = thickness//2 + padding

            bbox_ratio = (bbox_width*bbox_height)/image.shape[0]**2
            if (bbox_ratio < min_bbox_ratio):
                result, x_min, y_min, x_max, y_max = self.__crop_image(result, x_min, y_min, x_max, y_max, min_bbox_ratio)

            if is_red: 
                bbox_color = (255, 0, 0)
            else: 
                bbox_color = (0, 255, 255)
            
            x_min = max(margin, x_min - padding)
            y_min = max(margin, y_min - padding)
            x_max = min(w - 1 - margin, x_max + padding)
            y_max = min(h - 1 - margin, y_max + padding)

            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), bbox_color, thickness)

        return result

    def __stitch_bbox_images(
        self,
        images_list,
        max_images=8,
        bg_color=(0, 0, 0),
        gap=20,
        cell_size=512,
    ):

        from PIL import Image

        images = [img for img in images_list if img is not None][:max_images]
        n = len(images)
        if n == 0:
            return None

        images = [
            img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB")
            for img in images
        ]

        nrows = min(2, n)
        ncols = (n + nrows - 1) // nrows  # ceil

        tile_w = cell_size - gap
        tile_h = cell_size - gap
        if tile_w <= 0 or tile_h <= 0:
            raise ValueError("gap must be smaller than cell_size")

        pad_l = gap // 2
        pad_t = gap // 2

        images = [
            img.resize((tile_w, tile_h), resample=Image.BILINEAR) if img.size != (tile_w, tile_h) else img
            for img in images
        ]

        canvas_w = ncols * cell_size
        canvas_h = nrows * cell_size
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

        for i, img in enumerate(images):
            r = i // ncols
            c = i % ncols

            cell_x = c * cell_size
            cell_y = r * cell_size

            x = cell_x + pad_l
            y = cell_y + pad_t
            canvas.paste(img, (x, y))

        return canvas



if __name__ == '__main__':
    seq_name = 'disordered_A_Bathroom-18516'
    root_env_dir = '/workspace/data/scenes'
    data_root_dir = '/workspace/data/scenes/disordered/A/Bathroom-18516/frames'
    seg3D_path = '/workspace/data/scenes/disordered/A/Bathroom-18516/frames'
    save_root_path = '/workspace/data/scenes/disordered/A/Bathroom-18516/tasmap_pp_results'
    model_name = 'o4-mini-2025-04-16'   

    api_key = "YOUR_API_KEY_HERE"

    inferencer = TMPPInference(root_env_dir=root_env_dir,
                                seq_name=seq_name, 
                                seg3D_path=seg3D_path,
                                save_root_path=save_root_path,
                                model_name=model_name,
                                api_key=api_key,
                                debug=True)
    inferencer.run()
    inferencer.semantic_fusion()
    inferencer.visualize()
