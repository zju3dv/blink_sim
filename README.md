# BlinkSim
A versatile simulator for advancing research in event-based and RGB-event data fusion.
<br/>

## Demos

Automatically generated results where objects are randomly selected from a pool and then placed and moved according to some pre-defined rules (also the camera):

### Results of purely random scenes

V1 (i.e. BlinkFlow):

![Demo_Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/v1.gif?raw=true)

V2 (need to checkout dev/v2 branch):
![Demo Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/auto.gif?raw=true)


### Rendered result of customized scene:

Note: need to checkout dev/v2 branch

![Demo Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/custom.gif?raw=true)


## Features (some need to checkout dev/v2 branch)

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


## Installation

1. Install Blender, recommended version 3.3, link: https://www.blender.org/download/lts/3-3/
2. Install Python dependencies

```bash
conda env create -f environment.yml
```
3. Prepare data and put them under `data/`. The data includes:

```text
1. ADE20K dataset, or other image dataset that can be used as texture
2. ShapeNetCore.v2 dataset, or other 3D model dataset
3. hdri dataset, we provide a download script in scripts/download_hdri.py
```

We provide sample data for fast testing. You can download them using the following command:

```bash
python scripts/download_hf_data.py
```


4. (Optional) If you are running rendering on a headless machine, you will need to start an xserver. To do this, run:

```bash
sudo apt-get install xserver-xorg
sudo python3 scripts/start_xserver.py start
export DISPLAY=:0.{id} # for example, to use the GPU card 0, it should be DISPLAY=:0.0
```

5. Run the main script

If you want to use the default config (need the full dataset), you can run:
```bash
python main.py
```

Else if you want to use the sample data, you can run:
```bash
python main.py --config configs/blinkflow_v1_example.yaml
```

If it runs successfully, you will see the similar result under `output` folder:
```text
output/train/000000
├── events_left
├── forward_flow
├── hdr
└── hdr.mp4
```

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
```

```bibtex
@inproceedings{blinkvision_eccv2024,
  title={BlinkVision: A Benchmark for Optical Flow, Scene Flow and Point Tracking Estimation using RGB Frames and Events},
  author={Yijin Li, Yichen Shen, Zhaoyang Huang, Shuo Chen, Weikang Bian, Xiaoyu Shi, Fu-Yun Wang, Keqiang Sun, Hujun Bao, Zhaopeng Cui, Guofeng Zhang, Hongsheng Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

