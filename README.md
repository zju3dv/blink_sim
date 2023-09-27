# BlinkSim
A versatile simulator for advancing research in event-based and RGB-event data fusion.
<br/>

## Demos

Automatically generated results where objects are randomly selected from a pool and then placed and moved according to some pre-defined rules (also the camera):

![Demo Video](https://github.com/eugenelyj/open_access_assets/blob/master/blinkflow/auto.gif?raw=true)


Rendered result of customized scene:

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

## Related Benchmark & Training Data: [BlinkFlow](https://zju3dv.github.io/blinkflow/)

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

