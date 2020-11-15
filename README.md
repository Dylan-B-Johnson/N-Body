# N-Body
A Python N-Body Simulator With 3D Graphing Over Time (in GIF format).

### Info
The simulation uses the Verlet integration for the Particle-Particle N-Body algorithm. This implementation is O(n<sup>2</sup>). This high complexity means that the GPU really shouldn't be used, as the scales that would be required to get a benefit would take a very long time to run. The Particle Mesh algorithm is only O(n), so once it is implimented, the GPU can be used. There are four example outputs in this repository (for one of them the data was saved and plotted in GNUplot so trajectories could be seen).

### How to Use
1. Set arrays for x, v, and m (using one of the prebuilt methods or any other way).
2. Set data equal to a call of VerletCPU/GPU.
3. Choose the reporting frequency, graphing range, and output location and run the program.

### To Do
- Add the Barnes–Hut algorithm 
- Add the Particle Mesh algorithm
- Add the Particle-Particle Mesh algorithm
- Add a galaxy generator 
- Add Ruth 3rd-Order Integrator 
- Add Runge-Kutta 4th-Order Integrator 

### License:
Copyright 2020 Dylan Johnson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
