"""
Copyright [2020] [Dylan Johnson]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cupy as cp
from cupy import linalg as la
from numpy import linalg as la2
import matplotlib.animation as anime
import matplotlib.pyplot as plt
import time
from numba import jit

# This will be slower than on the CPU for all but very large simulations
# Use this after creating x, v, m, with another method
def verlet_gpu(x, v, m, sim_len, sim_iter, t=0.0, G=6.67408e-11):
	h = sim_len/sim_iter    
	n = int(x.shape[0])
	data_x = np.empty(((sim_iter+1),n,3))
	
	data_x[0,:,:]=x.get()
					  
	for i in range(1,(sim_iter+1)):
		v_old=v
		v+=calc_acc_gpu(x, m, n, G)*h
		x+=(v+v_old)*(0.5*h)
		data_x[i,:,:]=x.get()
	return data_x

# This will be faster in most scenarios 
# Use this after creating x, v, m, with another method
@jit(nopython=True)
def verlet_cpu(x, v, m, sim_len, sim_iter, t=0.0, G=6.67408e-11):
	h = sim_len/sim_iter    
	n = int(x.shape[0])
	data_x = np.empty(((sim_iter+1),n,3))
	
	data_x[0,:,:]=x
					  
	for i in range(1,(sim_iter+1)):
		v_old=v
		v+=calc_acc(x, m, n, G)*h
		x+=(v+v_old)*(0.5*h)
		data_x[i,:,:]=x
	return data_x

def calc_acc_gpu(x, m, n, G):
	a=cp.zeros((n,3))
	for i in range(n):
		for j in range(n):
				if j!=i:
					r_diff=x[j,:]-x[i,:]
					a[i,:]+=r_diff*G*m[j]/(la.norm(r_diff)**3.0)
	return a

@jit(nopython=True)
def calc_acc(x, m, n, G):
	a=np.zeros((n,3))
	for i in range(n):
		for j in range(n):
				if j!=i:
					r_diff=x[j,:]-x[i,:]
					a[i,:]+=r_diff*G*m[j]/(la2.norm(r_diff)**3.0)
	return a

def update(i, data_x, ax):
	global x_range
	global y_range
	global z_range
	global no_range
	global reporting_freq
	if i%reporting_freq==0:
		ax.clear()
		if not no_range:
			ax.axes.set_xlim3d(x_range[0],x_range[1])
			ax.axes.set_ylim3d(y_range[0],y_range[1])
			ax.axes.set_zlim3d(z_range[0],z_range[1])
		ax.scatter3D(data_x[i,:,0], data_x[i,:,1], data_x[i,:,2])

def animate(data_x):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	anim = anime.FuncAnimation(fig, update, frames=range(data_x.shape[0]), fargs=(data_x, ax))
	return anim

# generates n random bodies with mass on the order of m_order, a velocity on the order of v_order, and an x on the order of x_order
def random_m_x_v(n, m_order=1.0e10, v_order=0.0, x_order=1.0e10, gpu=False):
	global x
	global m
	global v
	if gpu:
		x=cp.random.randn(n,3)*x_order
		v=cp.random.randn(n,3)*v_order
		m=cp.random.randn(n)*m_order
	else:
		x=np.random.randn(n,3)*x_order
		v=np.random.randn(n,3)*v_order
		m=np.random.randn(n)*m_order

# generates a sphere with radius r, n bodies, a mass on the order of m_order, a velocity on the order of v_order
# If spheres_towards_point==True, then set point to the point the spheres velocity is towards
# If star==True, a star is placed at the center of the simulation (with mass star_multiplier times that of all other bodies in the simulation)
def sphere(n, r=8.0e11, m_order=1.0e34, v_order=1.0e5, inward_v=False, star=False, gpu=False, spheres_towards_point=False, point=[0.0,0.0,0.0], star_multiplier=1.0e2,
	spheres=[False, 'num_spheres (int)', '[[sphere 1 x,y,z offset],[sphere 2 x,y,z offset]], ect... (list of lists)']):
	global x
	global m
	global v
	if gpu:
		m=cp.random.randn(n)*m_order
		if not inward_v:
			v=cp.random.randn(n,3)*v_order
		phi=cp.random.randn(n)*cp.pi
		theta=abs(cp.random.randn(n))*2.0*cp.pi
		x=cp.empty((n,3))
		x[:,0] = r * cp.cos(theta) * cp.sin(phi)
		x[:,1] = r * cp.sin(theta) * cp.sin(phi)
		x[:,2] = r * cp.cos(phi)
		if star: 
			m[0]=m_order*star_multiplier*n
			x[0,:]=cp.array([0.0,0.0,0.0])
		if spheres[0]:
			if n%spheres[1]==0:
				section_size=int(n/spheres[1])
				last_index=0
				for sec in range(1,(spheres[1]+1)):
					offset=cp.array(spheres[2][(sec-1)])
					x[last_index:(section_size*sec),:]+=offset
					last_index=section_size*sec
				if star: 
					for i in range(spheres[1]):
						m[i*section_size]=m_order*star_multiplier*n
						offset=cp.array(spheres[2][(i)])
						star_r=cp.array([0.0,0.0,0.0])+offset
						x[i*section_size,:]=star_r
				if spheres_towards_point:
					for i in range(n):
						v*=(cp.array([0.0,0.0,0.0])-v)/(la.norm(cp.array(point)-v))
			else:
				print('WARNING: n must be divisible by the number of spheres.')
	else:
		m=np.random.randn(n)*m_order
		if not inward_v:
			v=np.random.randn(n,3)*v_order
		phi=np.random.randn(n)*np.pi
		theta=abs(np.random.randn(n))*2.0*np.pi
		x=np.empty((n,3))
		x[:,0] = r * np.cos(theta) * np.sin(phi)
		x[:,1] = r * np.sin(theta) * np.sin(phi)
		x[:,2] = r * np.cos(phi)
		if star: 
			m[0]=m_order*star_multiplier*n
			x[0,:]=np.array([0.0,0.0,0.0])
		if spheres[0]:
			if n%spheres[1]==0:
				section_size=int(n/spheres[1])
				last_index=0
				for sec in range(1,(spheres[1]+1)):
					offset=np.array(spheres[2][(sec-1)])
					x[last_index:(section_size*sec),:]+=offset
					last_index=section_size*sec
				if star: 
					for i in range(spheres[1]):
						m[i*section_size]=m_order*star_multiplier*n
						offset=np.array(spheres[2][(i)])
						star_r=np.array([0.0,0.0,0.0])+offset
						x[i*section_size,:]=star_r
				if spheres_towards_point:
					for i in range(n):
						v*=(np.array([0.0,0.0,0.0])-v)/(la2.norm(np.array(point)-v))
			else:
				print('WARNING: n must be divisible by the number of spheres.')
                
# make rest of simulation first 
# can do multiple times with different 
def big_mass(index,other_m_order=1.0e34, mass_multiplier=1.0e2,position=[0.0,0.0,0.0], gpu=False, velocity=[0.0,0.0,0.0]):
	global x
	global m
	global v
	if gpu:
		x[index,:]=cp.array(position)
		v[index,:]=cp.array(velocity)
	else:
		x[index,:]=np.array(position)
		v[index,:]=np.array(velocity)
	m[index]=mass_multiplier*len(m)*other_m_order

# Sets the initial conditions for the solar system 
def solar_system(gpu=False):
	global x
	global m
	global v
	if gpu:
		x = cp.array([[0.0,0.0,0],
			[0.0,5.0e10,0],
			[0.0,1.1e11,0],
			[0.0,1.5e11,1e10],
			[0.0,2.2e11,0],
			[0.0,7.7e11,0],
			[0.0,1.4e12,0],
			[0.0,2.8e12,0],
			[0.0,4.5e12,0],
			[0,3.7e12,0]])
		v = cp.array([[0,0,0],
			[47000,0,0],
			[35000,0,0],
			[30000,0,0],
			[24000,0,0],
			[13000,0,0],
			[9000,0,0],
			[6835.0e0,0,0],
			[5477.0e0,0,5.0],
			[4748.0e0,0,0]])
		m=cp.array([1.989e30,3.285e23,4.867e24,5.972e24,6.39e23,1.898e27,5.683e26,8.681e25,1.024e26,3.0e30])
	else:
		x = np.array([[0.0,0.0,0],
			[0.0,5.0e10,0],
			[0.0,1.1e11,0],
			[0.0,1.5e11,1e10],
			[0.0,2.2e11,0],
			[0.0,7.7e11,0],
			[0.0,1.4e12,0],
			[0.0,2.8e12,0],
			[0.0,4.5e12,0],
			[0,3.7e12,0]])
		v = np.array([[0,0,0],
			[47000,0,0],
			[35000,0,0],
			[30000,0,0],
			[24000,0,0],
			[13000,0,0],
			[9000,0,0],
			[6835.0e0,0,0],
			[5477.0e0,0,5.0],
			[4748.0e0,0,0]])
		m=np.array([1.989e30,3.285e23,4.867e24,5.972e24,6.39e23,1.898e27,5.683e26,8.681e25,1.024e26,3.0e30])

if __name__ == '__main__':
		# Creating Initial Conditions 
		start = time.time()
		sphere(1000,r=1.2e12, v_order=0.0,gpu=False, star=True, m_order=1.0e34)
		
		# Running Simulation
		data = verlet_cpu(x,v,m,5.8e5/24.0,360)
		print(str(time.time()-start))  

		# Plotting Results
		start = time.time()
		x_range=[-2e12,2e12]
		y_range=[-2e12,2e12]
		z_range=[-2e12,2e12]
		reporting_freq=1
		no_range=False
		anim = animate(data)
		anim.save('animation.gif', writer='pillow')
		print(str(time.time()-start))