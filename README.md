## 3. Particle Filter for Kidnapped Vehicle Project

### Self-Driving Car Engineer Nanodegree Program - Term 2

Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data. 

[//]: # (Image References)
[image1]: ./images/screenshot.png

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

INPUT: values provided by the simulator to the c++ program

```
// sense noisy position data from the simulator

["sense_x"] 

["sense_y"] 

["sense_theta"] 

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"] 

["sense_observations_y"] 

```

OUTPUT: values provided by the c++ program to the simulator
```
// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"] 

//Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label 

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions

```
Your job is to build out the methods in `particle_filter.cpp` until the simulator output says:

```
Success! Your particle filter passed!
```

The result looks like the following on the simulator:

![alt text][image1]

### Data to the Particle Filter
You can find the inputs to the particle filter in the `data` directory. 

`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### Basic Build Instructions

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_filter

### Code Style

Follow [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html) as much as possible.

