# Learning Predictive Vehicle-Terrain Interaction for Safe Off-Road Navigation

**A safe, efficient, and agile ground vehicle navigation algorithm for 3D off-road terrain environments.**
<<## Usage
This GitHub repository is currently under construction, and we will be updating it with comprehensive instructions on setting up the environment and running the code.
>>
## System architecture
<img src="https://github.com/HMCL-UNIST/Interaction-aware-3DOffroad/assets/32535170/3360407b-6669-4f22-9066-292ad76d356e" width="500">

We design a system that learns the terrain-induced uncertainties from driving data and encodes the learned uncertainty distribution into the
traversability cost for path evaluation. The navigation path is then designed to optimize the uncertainty-aware traversability cost, resulting in a safe and agile vehicle maneuver.  

## Dependency

Tested with ROS Noetic, torch 1.12.0+cu116, gpytorch 1.8.0 

1. To install Gazebo simulation environment
->  Please follow the installation tutorial in "https://github.com/AutoRally/autorally.git" 
2. For Mapping modules 
-> Dependencis can be found at "https://github.com/leggedrobotics/traversability_estimation.git"
3. NVIDIA Graphic card and Gpytorch is required 
```
pip install gpytorch
```
4. pytorch is required 
-> https://pytorch.org/get-started/locally/


## Install

Use the following commands to download and compile the package.
```
cd ~/catkin_ws/src
git clone https://github.com/HMCL-UNIST/Interaction-aware-3DOffroad.git 
cd ..
catkin build 
```

## Run the package

1. Run the Gazebo Simulation(e.g., environment with mud area) :
```
roslaunch autorally_gazebo mud_path.launch
```

2. Run the preprocessing Modules  :
```
roslaunch elevation_mapping elevation_mapping.launch
roslaunch traversability_estimation traversability_estimation.launch
```

3. Run the predictive vehicle-terrain interaction-aware path planning Module  :
```
roslaunch auc auc_main.launch
```

4. Run the Model predictive controller (MPPI) :
```
roslaunch mppi_ctrl mppi_ctrl.launch
```

5. Run low level controller 
```
roslaunch lowlevel_ctrl lowlevel_ctrl.launch
```



## Paper 
Hojin Lee, Sanghun Lee, and Cheolhyeon Kwon, Learning Predictive Vehicle-Terrain Interaction for Safe Off-Road Navigation
, Submitted to 2024 ICRA


## Acknowledgement
 **I would like to express my sincere thanks to following**
- Our 3D Simulation environment and the Gazebo vehicle model is based on Autorally research platform  
```
(Goldfain, Brian, et al. "Autorally: An open platform for aggressive autonomous driving." IEEE Control Systems Magazine 39.1 (2019): 26-55.)  
```

- Elevation and traversability mapping modules used in preprocessing step are based on these awesome work. 
```
( P. Fankhauser, M. Bloesch, C. Gehring, M. Hutter, and R. Siegwart,
        “Robot-centric elevation mapping with uncertainty estimates,” in Mobile
          Service Robotics. World Scientific, 2014, pp. 433–440.) 
```       

```
(P. Fankhauser, M. Bloesch, and M. Hutter, “Probabilistic terrain
mapping for mobile robots with uncertain localization,” IEEE Robotics
and Automation Letters, vol. 3, no. 4, pp. 3019–3026, 2018.) 
```
