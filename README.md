# BlinkSim
A versatile simulator for advancing research in event-based and RGB-event data fusion.
<br/>

## Demos

Automatically generated results where objects are randomly selected from a pool and then placed and moved according to some pre-defined rules (also the camera):

### Results of purely random scenes

V2 (need to checkout dev/v2 branch):

![Demo Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/auto.gif?raw=true)


### Rendered result of customized scene (still not available):

![Demo Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/custom.gif?raw=true)


## Features

- Event simulation: event data simulated from high-frequency rendering data
- Simulation of low dynamic range, motion blur, defocus blur and atmospheric effect
- Dense point tracking: provide tracking ground truth for each pixel at any frame and any object
- Forward/backward optical flow
- Depth maps

Datas that are not shown in the demo but are also accessible

- Normal maps
- Instance segmentation
- Camera poses and intrinsic
- Object poses

## Related Benchmark & Training Data:

1. [BlinkFlow](https://zju3dv.github.io/blinkflow/)
2. [BlinkVision](https://www.blinkvision.net/)


## Installation and Usage

1. Install Blender, recommended version 3.3, link: https://www.blender.org/download/lts/3-3/
2. Install Python dependencies

```bash
conda env create -f environment.yml
```
3. Prepare data and put them under `data/`. The full data that we used includes:

```markdown
**texture**
1. ADE20K dataset, need to download yourself
2. flickr images
3. pixabay images
4. cc textures

**object**
1. shapenet dataset, need to download yourself
2. google scanned dataset, need to download yourself
```

We crawled some images for usage such as texture and HDR lighting, including flickr, pixabay and cc textures and so on.
Download our prepared data through:

```bash
python scripts/download_hf_data_v2.py
```

We also provide sample data for fast testing. You can download them using the following command:

```bash
python scripts/download_hf_data_example.py
```


4. (Optional) If you are running rendering on a headless machine, you will need to start an xserver. To do this, run:

```bash
sudo apt-get install xserver-xorg
sudo python3 scripts/start_xserver.py start
export DISPLAY=:0.{id} # for example, to use the GPU card 0, it should be DISPLAY=:0.0
```

5. Run the main script

If you want to use the default config (need to prepare full dataset), you can run:
```bash
python main.py
```

Else if you want to use the sample data, you can run:
```bash
python main.py --config configs/blinkflow_v2_example.yaml
```

If it runs successfully, you will see the similar result under `output` folder:
```text
output/train/000000
├── events_left
├── forward_flow
├── clean_uint8
├── all_instance.txt
├── dynamic_instance.txt
├── event_ts.txt
├── image_ts.txt
├── metadata.json
└── clean.mp4
```

In the default config, we disable the rendering and parsing of many data such as stereo data and the groud truth of particle tracking, depth and so on. You can refer to the config (configs/blinkflow_v2.yaml) and enable them if you need.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{blinkflow_iros2023,
  title={BlinkFlow: A Dataset to Push the Limits of Event-based Optical Flow Estimation},
  author={Yijin Li, Zhaoyang Huang, Shuo Chen, Xiaoyu Shi, Hongsheng Li, Hujun Bao, Zhaopeng Cui, Guofeng Zhang},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  month = {October},
  year = {2023},
}

@inproceedings{blinkvision_eccv2024,
  title={BlinkVision: A Benchmark for Optical Flow, Scene Flow and Point Tracking Estimation using RGB Frames and Events},
  author={Yijin Li, Yichen Shen, Zhaoyang Huang, Shuo Chen, Weikang Bian, Xiaoyu Shi, Fu-Yun Wang, Keqiang Sun, Hujun Bao, Zhaopeng Cui, Guofeng Zhang, Hongsheng Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

