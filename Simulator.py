import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import re
import random
import sys, getopt
import os

import matplotlib
import itertools 
import matplotlib.cm as cm
import pickle
import sklearn

from sklearn.linear_model import LogisticRegression as logis
from sklearn.metrics import confusion_matrix
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import mixture

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
    

'''
Simulator to create uv-tracks, add noise and RFI, and to detect RFI using 
machine learning.
'''


class sim():

      '''
      Constructor

      RETURNS:
      None

      LIST OF PARAMETERS:
      Dec_deg/ Dec_rad - Declination of observation in degree or radians
      L_deg/ L_rad - Latitude of array in degree or radians
      f - Frequency of observation in Hz
             #freq = 1.4*10**9 =1.4 GHz
      c - Speed of light
      Lam - Wavelength
      H_start = Starting hour angle
      H_stop = Stopping hour angle
      H_rad= np.linspace(H_start, H_stop, time_steps)  #Hour angle window
      time_steps - Number of timesteps
      D - baseline length
      A - azimuth angle of baseline
      E - elevation angle of baseline
      L_rad - latitude of array
      b - baseline
      '''

      """
      Loading the antenna layout file and the observation parameters
      """
      enu_data = pd.read_excel ('C:/Users/insig_000/Desktop/ENU coordinates of KAT-7.xlsx', index_col = 0) ## Put file path here.

      ### Cleaning the data a bit ###
      enu_data.columns = ['x','y','z'] # Renaming columns for convenience
      enu_data['x'] = enu_data['x'].apply(lambda x: x.replace('m', '').replace('(x1)', '')).astype('float')
      enu_data['y'] = enu_data['y'].apply(lambda x: x.replace('m', '').replace('(y1)', '')).astype('float')
      enu_data['z'] = enu_data['z'].apply(lambda x: x.replace('m', '').replace('(z1)', '')).astype('float')
      enu_data.head()
      print(enu_data)
  
      enu_data1 = pd.read_excel ('C:/Users/insig_000/Desktop/Additional Info.xlsx', index_col = 0) ## Put file path here.
      print(enu_data1)
     

      def __init__(self,nsteps=100,c=3e8):
          self.N = 0 #Number of antennas
          self.B = 0 #Number of baselines
          self.nsteps = nsteps
          self.c = c #Speed of light
          self.ant = np.array([])
          self.u_m = None #The uvw composite index matrices
          self.v_m = None
          self.w_m = None
          

      def latitude(self):
          L = (self.enu_data1.iloc[0,0])
          L = re.findall("[-0-9.eE]+", L)
          L = float(L[0]), float (L[1]), float (L[2])
          L_deg = (L[0] + (L[1]/60) + (L[2]/3600))  
          L_rad = (np.pi/180)*L_deg
          return L_rad


      def declination(self):
          Dec = (self.enu_data1.iloc[3,0])
          Dec = re.findall("[-0-9.eE]+", Dec)
          Dec = float(Dec[0]), float (Dec[1]), float (Dec[2])
          Dec_deg = (Dec[0] + (Dec[1]/60) + (Dec[2]/3600)) 
          Dec_rad = (np.pi/180)*Dec_deg
          return Dec_rad


      def lam(self):
          f = (self.enu_data1.iloc[5,0])                                                 
          Lam = self.c/f 
          return Lam


      def hour_angle_range(self):                             
          H_start = (self.enu_data1.iloc[1,0])
          H_start = re.findall("[-0-9.eE]+", H_start)
          self.H_start = float(H_start[0])*np.pi/12 

          H_stop = (self.enu_data1.iloc[2,0])
          H_stop = re.findall("[-0-9.eE]+", H_stop)
          self.H_stop = float(H_stop[0])*np.pi/12 
                                         
          H_rad = np.linspace(self.H_start, self.H_stop, self.nsteps)           
          return H_rad


      def antenna_positions(self):
          self.ant = np.array(self.enu_data.iloc[:,:])
          return self.ant  


      def num_antennas(self):
          self.N=self.ant.shape[0]
          return self.N


      def num_baselines(self):
          self.B = int((self.N**2-self.N)/2)
          return self.B


      '''
      Converts baseline length, azimuth angle, elevation angle and latitude into XYZ coordinates
      INPUTS:
            D - baseline length
            A - azimuth angle of baseline
            E - elevation angle of baseline
            L_rad - latitude of array
      RETURNS:
            XYZ - a vector of size 3 containing the XYZ coordinate of a specific baseline 
      '''


      def DAE_to_XYZ(self,D,A,E,L_rad):
          XYZ = D * np.array([np.cos(L_rad)*np.sin(E) - np.sin(L_rad)*np.cos(E)*np.cos(A),
          np.cos(E)*np.sin(A),
          np.sin(L_rad)*np.sin(E) + np.cos(L_rad)*np.cos(E)*np.cos(A)])
          return XYZ


      '''
      Computes the rotation matrix A needed to convert XYZ into uvw at a specific hour angle        
      RETURNs:
      A - rotation matrix capable of converting XYZ into uvw
      '''

      def XYZ_to_uvw(self,H_rad,delta):
          A = np.array([[np.sin(H_rad),np.cos(H_rad),0],[-1*np.sin(delta)*
          np.cos(H_rad),np.sin(delta)*np.sin(H_rad),np.cos(delta)],
          [np.cos(delta)*np.cos(H_rad),-1*np.cos(delta)*np.sin(H_rad),
          np.sin(delta)]])
          return A   


      def uv_track(self,H_rad,d,az,el,L_rad,Dec_rad,Lam):
          uvw = np.zeros((len(self.H_rad),3))
          for i in range(len(H_rad)):
              A = self.XYZ_to_uvw(H_rad[i],Dec_rad)
              uvw[i,:] = A.dot(self.DAE_to_XYZ(d,az,el,L_rad)/Lam)
          return uvw 

      
      def uv_tracks(self):
          DAE = np.zeros((self.B,3))#storing baseline length, azimuth angle and elevation
          
          p = np.zeros((self.B,),dtype=int) 
          q = np.zeros((self.B,),dtype=int)  

          k = 0
          for i in range(self.N):
              for j in range(i+1,self.N):
                  DAE[k,0] = np.sqrt((self.ant[i,0]-self.ant[j,0])**2+(self.ant[i,1]-self.ant[j,1])**2+(self.ant[i,2]-self.ant[j,2])**2)   
                  DAE[k,1] = np.arctan2((self.ant[j,0]-self.ant[i,0]), (self.ant[j,1]-self.ant[i,1])) 
                  DAE[k,2] = np.arcsin((self.ant[j,2]-self.ant[i,2])/DAE[k,0])
                  p[k] = i
                  q[k] = j
                  k = k + 1 

          #CONVERT TO UVW
          self.u_m = np.zeros((self.N, self.N, self.nsteps),dtype=float) 
          self.v_m = np.zeros((self.N, self.N, self.nsteps),dtype=float)
          self.w_m = np.zeros((self.N, self.N, self.nsteps),dtype=float)    
          
         
          for i in range(self.B):
              self.H_rad = self.hour_angle_range()  
              self.L_rad = self.latitude()
              self.Dec_rad = self.declination()
              self.Lam = self.lam()
              X = self.uv_track(self.H_rad,DAE[i,0],DAE[i,1],DAE[i,2],self.L_rad,self.Dec_rad,self.Lam)
              self.u_m[p[i],q[i],:] = X[:,0]
              self.u_m[q[i],p[i],:] = -1*X[:,0]
              self.v_m[p[i],q[i],:] = X[:,1]
              self.v_m[q[i],p[i],:] = -1*X[:,1] 
              self.w_m[p[i],q[i],:] = X[:,2]
              self.w_m[q[i],p[i],:] = -1*X[:,2]
              
      '''      
      Plots the uv-tracks of an array layout
      RETURNS:
      None
      ''' 

      def plot_uv_coverage(self):
          m_x = np.amax(np.absolute(self.u_m))
          m_y = np.amax(np.absolute(self.v_m)) 
          for i in range(self.N):
              for j in range(i+1,self.N):
                  plt.plot(self.u_m[i,j,:],self.v_m[i,j,:],"b")  
                  plt.plot(self.u_m[j,i,:],self.v_m[j,i,:],"r")
          plt.ylim(-1.1*m_y,1.1*m_y)
          plt.xlim(-1.1*m_x,1.1*m_x)
          plt.xlabel("$u$ [rad$^{-1}$]",fontsize=12)
          plt.ylabel("$v$ [rad$^{-1}$]", fontsize=12)
          plt.grid(color='lightgray',linestyle='--')
          plt.title("$UV$-Coverage for all Baselines", fontsize=12)
          #plt.show()
          #plt.savefig("uv_cov.png",bbox_inches='tight')
          plt.clf()
             

      '''      
      Plots a uv-track of a single baseline
      ''' 

      def single_baseline(self):
          DAE = np.zeros((self.B,3))#storing baseline length, azimuth angle and elevation
          
          p = np.zeros((self.B,),dtype=int) 
          q = np.zeros((self.B,),dtype=int)  

          k = 0
          for i in range(self.N):
              for j in range(i+1,self.N):
                  DAE[k,0] = np.sqrt((self.ant[i,0]-self.ant[j,0])**2+(self.ant[i,1]-self.ant[j,1])**2+(self.ant[i,2]-self.ant[j,2])**2)   
                  DAE[k,1] = np.arctan2((self.ant[j,0]-self.ant[i,0]), (self.ant[j,1]-self.ant[i,1])) 
                  DAE[k,2] = np.arcsin((self.ant[j,2]-self.ant[i,2])/DAE[k,0])
                  p[k] = i
                  q[k] = j
                  k = k + 1 
    

          for i in range(self.B):
              X = self.uv_track(self.H_rad,DAE[0,0],DAE[0,1],DAE[0,2],self.L_rad,self.Dec_rad,self.Lam) 
              self.u_m[p[i],q[i],:] = X[:,0]
              self.u_m[q[i],p[i],:] = -1*X[:,0]
              self.v_m[p[i],q[i],:] = X[:,1]
              self.v_m[q[i],p[i],:] = -1*X[:,1] 
              self.w_m[p[i],q[i],:] = X[:,2]
              self.w_m[q[i],p[i],:] = -1*X[:,2]

              self.D = DAE[0,0]
              self.A = DAE[0,1]
              self.E = DAE[0,2]
              self.u_coord = X[:,0]
              self.v_coord = X[:,1]
          return self.u_coord, self.v_coord
         

      def plot_uv_track(self):
          m_x = np.amax(np.absolute(self.u_m))
          m_y = np.amax(np.absolute(self.v_m)) 
          for i in range(self.N):
              for j in range(i+1,self.N):
                  plt.plot(self.u_m[i,j,:],self.v_m[i,j,:],"b")  
                  plt.plot(self.u_m[j,i,:],self.v_m[j,i,:],"r")
          plt.ylim(-1.1*m_y,1.1*m_y)
          plt.xlim(-1.1*m_x,1.1*m_x)
          plt.xlabel("$u$ [rad$^{-1}$]", fontsize=12)
          plt.ylabel("$v$ [rad$^{-1}$]", fontsize=12)
          plt.grid(color='lightgray',linestyle='--')
          plt.title("$UV$ track for a single baseline", fontsize=12)
          #plt.savefig("uv_track.png",bbox_inches='tight')
          #plt.show()
          plt.clf()



      '''
      Creating new sources
      point_sources - a point sources array with dimension num_sources x 3, with the second dimension denoting flux, l_0 and m_0 (in degrees) 
      respectively.
      a=Pareto parameter
      num_sources=number of sources
      fov=Field of view in which the sources reside (in degrees)
      '''
      
      def generate_flux(self):
          y = np.random.pareto(a=pareto_number, size=num_sources)
          return y         

      def generate_pos(self):
          return np.random.uniform(low=-1*np.absolute(fov), high=np.absolute(fov), size = num_sources)*(np.pi/180)
          
          
      def create_point_sources(self):
          self.point_sources = np.zeros((num_sources,3),dtype=float)  
          self.point_sources[:,0] = self.generate_flux()
          self.point_sources[:,1] = self.generate_pos()
          self.point_sources[:,2] = self.generate_pos()
          return self.point_sources   


      def baseLength_to_XYZ(self):
          self.X_singleBL = self.D*(np.cos(self.L_rad)*np.sin(self.E)-np.sin(self.L_rad)*np.cos(self.E)*np.cos(self.A))
          self.Y_singleBL = self.D*np.cos(self.E)*np.sin(self.A)
          self.Z_singleBL = self.D*(np.sin(self.L_rad)*np.sin(self.E)+np.cos(self.L_rad)*np.cos(self.E)*np.cos(self.A))      



      
      '''
      Creating visibilities
      '''

      def create_vis(self):
          u = np.linspace(-1*(np.amax(np.abs(self.u_coord)))-10, np.amax(np.abs(self.u_coord))+10, num=200, endpoint=True)
          v = np.linspace(-1*(np.amax(abs(self.v_coord)))-10, np.amax(abs(self.v_coord))+10, num=200, endpoint=True)   
          uu, vv = np.meshgrid(u, v)
          
          #creating a completely filled in  uv -plane and sample it on the EW-baseline track we created in the first section
          #Creating a fully-filled uv-plane
          #Setting frequency range
          f_start = 1.40E+09   
          f_stop = 1.90E+09
          self.time_steps = 200
          H = np.linspace(self.H_start, self.H_stop, self.time_steps)*(np.pi/12) #Hour angle in radians                                                                   
    
          #Creating array of frequency and time
          f = np.linspace(f_start, f_stop, self.time_steps)
          Lam = self.c/f 
          HH, ff = np.meshgrid(H, Lam)

          vv = np.zeros(HH.shape, dtype=float)
          uu = np.zeros(HH.shape, dtype=float)

          uu = ff**(-1)*(np.sin(HH)*self.X_singleBL + np.cos(HH)*self.Y_singleBL)
          vv = ff**(-1)*(-np.sin(self.Dec_rad)*np.cos(HH)*self.X_singleBL + np.sin(self.Dec_rad)*np.sin(HH)*self.Y_singleBL + np.cos(self.Dec_rad)*self.Z_singleBL)

          zz = np.zeros(uu.shape).astype(complex)
 
          s = self.point_sources.shape
          for counter in range(1, s[0]+1):
              A_i = self.point_sources[counter-1,0]
              l_i = self.point_sources[counter-1,1]
              m_i = self.point_sources[counter-1,2]
              zz += A_i*np.exp(-2*np.pi*1j*(uu*l_i+vv*m_i))
          self.zz = zz[:,::-1]


          '''
          Plotting the real and imaginary parts of visibilities that have been created 
          (100 sources, pareto number of 2 and FoV value of 10)
          '''
          plt.subplot(121)
          plt.imshow(self.zz.real,extent=[-1*(np.amax(np.abs(self.u_coord)))-10, np.amax(np.abs(self.u_coord))+10,-1*(np.amax(abs(self.v_coord)))-10, \
                           np.amax(abs(self.v_coord))+10])
          plt.plot(self.u_coord,self.v_coord,"k")
          plt.xlim([-1*(np.amax(np.abs(self.u_coord)))-10, np.amax(np.abs(self.u_coord))+10])
          plt.ylim(-1*(np.amax(abs(self.v_coord)))-10, np.amax(abs(self.v_coord))+10)
          plt.subplots_adjust(right=1.4)
          #plt.tight_layout()
          plt.xlabel("u", fontsize=12)
          plt.ylabel("v", fontsize=12)
          plt.title("Real part of visibilities", fontsize=12)

          plt.subplot(122)
          plt.imshow(self.zz.imag,extent=[-1*(np.amax(np.abs(self.u_coord)))-10, np.amax(np.abs(self.u_coord))+10,-1*(np.amax(abs(self.v_coord)))-10, \
                           np.amax(abs(self.v_coord))+10])
          plt.plot(self.u_coord,self.v_coord,"k")
          plt.xlim([-1*(np.amax(np.abs(self.u_coord)))-10, np.amax(np.abs(self.u_coord))+10])
          plt.ylim(-1*(np.amax(abs(self.v_coord)))-10, np.amax(abs(self.v_coord))+10)
          plt.xlabel("u", fontsize=12)
          plt.ylabel("v", fontsize=12)
          plt.title("Imaginary part of visibilities", fontsize=12)
          #plt.show()
          #plt.savefig("imag_and_real.png",bbox_inches='tight')
          plt.clf()


          #Computing visibilities for sources
          u_track = self.u_coord
          v_track = self.v_coord
          z = np.zeros(u_track.shape).astype(complex) 

          s = self.point_sources.shape
          for counter in range(1, s[0]+1):
              A_i = self.point_sources[counter-1,0]
              l_i = self.point_sources[counter-1,1]
              m_i = self.point_sources[counter-1,2]
              z += A_i*np.exp(-1*2*np.pi*1j*(u_track*l_i+v_track*m_i))


          '''
          Sampled visibilities with their real and imaginary components 
          (100 sources, pareto number of 2 and FoV value of 10)
          '''

          #Plotting the sampled visibilites as a function of time-slots
          plt.subplot(121)
          plt.plot(np.abs(z))
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Jy", fontsize=12)
          plt.title("Real: sampled visibilities", fontsize=12)

          plt.subplot(122)
          plt.plot(np.angle(z))
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Jy", fontsize=12)
          plt.title("Imag: sampled visibilities", fontsize=12)
          #plt.show()
          #plt.savefig("sampled.png",bbox_inches='tight')
          plt.clf()


          '''
          Plotting the phase and amplitude of the visibilities in a waterfall plot 
          (Baseline 12 - Antenna 1 and 2) as a function of frequency and timeslots. 
          '''
          #Plotting amplitude
          plt.subplot(311)
          #Amp = plt.imshow(zz.real, aspect = "auto")
          Amp = plt.imshow(np.abs(zz), aspect = "auto")
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          #plt.title("Visibilities of " + str(num_sources) + " created sources")
          plt.title("Visibilities", fontsize=12)
          plt.tight_layout(pad=0.4, w_pad=5.0, h_pad=2.0)
          plt.colorbar(Amp)
          plt.subplots_adjust(hspace=.0)
          ax = plt.gca()
          ax.axes.xaxis.set_visible(False)
          ax.axes.yaxis.set_ticks([])


          #Plotting phase
          plt.subplot(312)       #Plots one row and two columns,second subplot 
          #Phase=plt.imshow(zz.imag, aspect = "auto")
          Phase=plt.imshow(np.angle(zz), aspect = "auto")
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          #plt.title("Phase of visibilities", fontsize=12)
          plt.colorbar(Phase)
          plt.subplots_adjust(hspace=.0)
          #ax.set_xticklabels([])
          ax1 = plt.gca()
          ax1.axes.yaxis.set_ticks([])
          plt.show()
          #plt.savefig("visibilities.png",bbox_inches='tight')
          plt.clf()





 #__________________________________ADDITION OF NOISE_______________________________________ 
      '''
      Creating mask for waterfall, adding noise to sources with specific SNR value
      The lower the SNR value, the more noise that is added/higher the noise content
      #Mostly ranges from -10 to 10
      M[x] - Represents indices whose rows would be converted to ones--
      '''
      def mask(self):
          self.M = np.zeros((self.time_steps,self.time_steps),dtype=int) 
          randomlist = random.sample(range(0, 200), num_channels_to_corrupt)
          for i in randomlist:
              x=randomlist
              self.M[x]=1 
          #print("corrupted data=",self.M[x]) 
          #print("whole dataset=", self.M) 
          return self.M  
      


      ''''
      Determines the power in signal
      INPUTS:
      D - Matrix to calculate the power of
      RETURNS:
      D_pow - The power in each baseline
      d_pow1 - Average power over entire matrix
      '''     
                                                                
      def det_power_of_signal(self):
          D = np.copy(self.zz)
          D_pow = np.mean(np.absolute(D)**2,axis=1)
          d_pow1 = np.sum(np.absolute(D)**2)/(D.shape[0]*D.shape[1])                  
          return D_pow,d_pow1
    

      '''    
      Function that generates noise at a certain power level
      INPUTS:
      power - power of the noise which is generated from power needed for SNR
      '''      

      def generate_noise(self,power,second_dim=None):  #Read in entire D matrix
          sig = np.sqrt(power/2) 
          second_dim = None
          if second_dim is None:
             second_dim = self.time_steps 
          mat = sig*np.random.randn(second_dim)+sig*np.random.randn(second_dim)*1j
          return mat


      '''    
      Function that tells how much power in the noise is needed to achieve SNR
      P_signal - Power in the signal, SNR - SNR to achieve, P_noise - Power in the noise
      '''      
      def power_needed_for_SNR(self,P_signal,SNR):
          P_noise = P_signal*10**(-1*(SNR/10.))
          return P_noise  #Read into generate noise


      '''    
      Function that adds noise to visibilities of a certain SNR and to graph a waterfall plot
      P_signal - Power in the signal, SNR - SNR to achieve, P_noise - Power in the noise
      ''' 
      def adding_noise(self):
          if SNR is not None:
             D = np.copy(self.zz)
             d_pow1,P_signal = self.det_power_of_signal()
             P_noise = self.power_needed_for_SNR(P_signal,SNR)
             N = self.generate_noise(P_noise,second_dim=D.shape[1]) #Two dimensions
             D = D + N   #New matrix/visibility fxn
             sig = (np.sqrt(P_noise/2))


          '''
          Plotting a waterfall plot of the visibilities with thermal noise added.
          '''
          #Plotting Amplitude
          plt.subplot(311)
          Amp = plt.imshow(abs(D), aspect = "auto")           #Amplitude
          plt.ylabel("Frequency", fontsize=12)
          plt.title("Waterfall with thermal noise of SNR = " +str(SNR), fontsize=12)
          plt.tight_layout(pad=0.4, w_pad=5.0, h_pad=1.0)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax = plt.gca()
          ax.axes.xaxis.set_visible(False)
          ax.axes.yaxis.set_ticks([])

        
          #Plotting Phase
          plt.subplot(312)       #Plots one row and two columns,second subplot 
          Phase=plt.imshow(np.angle(D), aspect = "auto")      #Phase
          plt.ylabel("Frequency", fontsize=12)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax1 = plt.gca()
          ax1.axes.xaxis.set_visible(False)
          ax1.axes.yaxis.set_ticks([])
        
        
          #Plotting the corruption mask
          plt.subplot(313)
          plt.imshow(self.M, aspect = "auto")
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax2 = plt.gca()
          ax2.axes.yaxis.set_ticks([])
          ax2.axes.xaxis.set_ticks([])
          plt.show()
          #plt.savefig("waterfall_snr.png",bbox_inches='tight')
          plt.clf()




#__________________________________ADDITION OF RFI NOISE_______________________________________ 
      '''    
      Function that adds RFI to visibilities of a certain SNR and to graph a waterfall plot with both noise and RFI added.
      P_signal - Power in the signal, SNR - SNR to achieve, P_noise - Power in the noise
      #The lower the SNR value, the more noise that is added/higher the noise content
      ''' 
      def power_needed_for_SNR(self,P_signal_RFI, SNR_high):
          P_noise_RFI = P_signal_RFI*10**(-1*(SNR_high/10.))
          return P_noise_RFI  #Read into generate noise

      def adding_RFI(self):
      #Adding RFI Noise
          if SNR_high is not None:
             D_RFI = np.copy(self.zz)
             d_pow1,P_signal_RFI = self.det_power_of_signal()
             P_noise_RFI = self.power_needed_for_SNR(P_signal_RFI,SNR_high)
             N_RFI = self.generate_noise(P_noise_RFI,second_dim=D_RFI.shape[1])
             self.D_tot = D_RFI + self.N + N_RFI * self.M    
  

          '''
          Plotting a waterfall plot of the visibilities with RFI noise added.
          '''
          plt.subplot(311)
          Amp = plt.imshow(abs(self.D_tot), aspect = "auto")
          #plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          plt.title("Waterfall Plot with RFI effects added, RFI SNR = " +str(SNR_high), fontsize=12)
          plt.tight_layout(pad=0.4, w_pad=5.0, h_pad=1.0)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax = plt.gca()
          ax.axes.xaxis.set_visible(False)
          ax.axes.yaxis.set_ticks([])
          ax.axes.xaxis.set_ticks([])
        
        
          plt.subplot(312)       #Plots one row and two columns,second subplot 
          Phase=plt.imshow(np.angle(self.D_tot), aspect = "auto")
          #plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          #plt.title("Phase of visibilities", fontsize=12)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax1 = plt.gca()
          ax1.axes.xaxis.set_visible(False)
          ax1.axes.xaxis.set_ticks([])
          ax1.axes.yaxis.set_ticks([])
        
          plt.subplot(313)
          plt.imshow(self.M, aspect = "auto")
          plt.xlabel("Timeslots", fontsize=12)
          plt.ylabel("Frequency", fontsize=12)
          plt.colorbar()
          plt.subplots_adjust(hspace=.030)
          ax2 = plt.gca()
          ax2.axes.yaxis.set_ticks([])
          ax2.axes.xaxis.set_visible(True)
          ax2.axes.xaxis.set_ticks([])
          plt.show() 
          #plt.savefig("waterfall_rfi.png",bbox_inches='tight') 
          plt.clf()        





#__________________________________MACHINE LEARNING/CLASSIFICATION_______________________________________ 
      #CLASSIFICATION      
      def create_colour_maps(self):
          self.cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
          self.cmap_bold = ListedColormap(['#FF0000', '#00FF00']) 


      def plot_confusion_matrix(self, cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
          "    This function prints and plots the confusion matrix.\n",
          "    Normalization can be applied by setting `normalize=True`.\n",
    
    
          #Computing confusion matrix 
          #cm = confusion_matrix(y_true, y_pred)
          # Only use the labels that appear in the data
          if normalize:
             self.cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
             print("Normalized confusion matrix")
          else:
             print('Confusion matrix, without normalization')
          print("cm =", self.cm)
    
          plt.imshow(self.cm, interpolation='nearest', cmap=cmap),
          plt.title(title, fontsize=12)
          plt.colorbar()
          
          tick_marks = np.arange(len(classes))
          plt.xticks(tick_marks, classes, rotation=45)
          plt.yticks(tick_marks, classes)
          fmt = '.2f' if normalize else 'd'
          
          for i, j in itertools.product(range(2), range(2)):
              plt.text(j, i, format(self.cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white")
          plt.tight_layout()
          plt.ylabel('True label', fontsize=12)
          plt.xlabel('Predicted label', fontsize=12)
          plt.show()

          for i in range(number_of_experiment):
              path = type_classifier+"_"+"Experiments"
              image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
              image_folder = os.path.join(path,image_directory)

              if not os.path.isdir("./"+image_folder):
                 os.system("mkdir "+image_folder) 
          i=1
          while os.path.exists("./"+image_folder+"/" +type_classifier +f" CM_Image {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
              i+=1

          #plt.savefig("./"+image_folder+"/" +type_classifier +f" CM_Image {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png",bbox_inches='tight')
          plt.clf() 

      
      '''
      Setting the labels and splitting data into training and testing set
      ''' 
      def classification_parameters(self):
          self.y = self.M.flatten()
          #print(self.y)
          self.c = np.array(["r","g"])
          self.h = 0.02   
  
          #print("corrupted data=",self.M[x]) 
          #print("whole dataset with corrupted data=", self.M)   
 
          self.X = np.zeros((len(self.y),2),dtype=float)       
          self.X[:,0]=np.abs(self.D_tot).flatten()
          self.X[:,1]=np.angle(self.D_tot).flatten()
          self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.50, stratify=self.y)
          #print (len(self.X_train))


      '''
      Plots a scatter plot of true visibilities (red) and RFI (green)
      '''
      def scattered_plot(self): 
          #Generate scatter plot for training data 
          for k in range (len(self.y)):
              #if k%10==0:
              if k%100==0 or self.y[k]==1:
              	 plt.scatter(self.X[k,0],self.X[k,1],c=self.c[self.y[k]])

          plt.title('Scatter Plot', fontsize=12)
          plt.xlabel("Amplitude", fontsize=12)
          plt.ylabel("Phase", fontsize=12)
          plt.axis('tight')
          plt.xlim([0,250])
          plt.ylim([-4,4])
          plt.show()
          for i in range(number_of_experiment):
                 path = type_classifier+"_"+"Experiments"
                 image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
                 image_folder = os.path.join(path,image_directory)

                 if not os.path.isdir("./"+image_folder):
                    os.system("mkdir "+image_folder) 
          i=1
          while os.path.exists("./"+image_folder+"/" +type_classifier +f" Scattered_plot {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
              i+=1

          #plt.savefig("./"+image_folder+"/" +type_classifier +f" Scattered_plot {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png", bbox_inches='tight')
             
          plt.clf()
         
      

      def Naive_Bayes_Classifier(self):
          ## Training
          GaussianNB(priors=None)
          
          clf = GaussianNB()
          clf.fit(self.X_train, self.y_train)
          y_pred = clf.predict(self.X_test)
      
          # Plot the decision boundary. For that, we will assign a color to each\n",
          # point in the mesh [x_min, x_max]x[y_min, y_max].\n",
          #Define grid of points by finding the minimum and maximum values for each feature and expanding grid one step beyond.
          x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
          y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                               np.arange(y_min, y_max, self.h))
          Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
          # Put the result into a color plot\n",
          Z = Z.reshape(xx.shape)

          if type_classifier == 'Naive':
             plt.pcolormesh(xx, yy, Z, shading='auto', cmap=self.cmap_light)
             # Plot also the training points\n",
             plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap=self.cmap_bold,
                   edgecolor='k', s=20)
             plt.title("Phase vs Amplitude (Naive Bayes)", fontsize=12)
             plt.axis('tight')
             plt.xlabel('Amplitude', fontsize=12)
             plt.ylabel('Phase', fontsize=12)
             plt.xlim([0,250])
             plt.ylim([-4,4])
             plt.show()
             for i in range(number_of_experiment):
                 path = type_classifier+"_"+"Experiments"
                 image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
                 image_folder = os.path.join(path,image_directory)

                 if not os.path.isdir("./"+image_folder):
                    os.system("mkdir "+image_folder) 
             i=1
             while os.path.exists("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
                 i+=1

             #plt.savefig("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png",bbox_inches='tight')
             plt.clf()

             cm = confusion_matrix(self.y_test,y_pred)
             self.plot_confusion_matrix(cm,self.c)
             

  
      def Logistic_Regression(self):
      ## Training
          clf = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
          clf.fit(self.X_train, self.y_train)
          y_pred = clf.predict(self.X_test)
     
          # Plot the decision boundary. For that, we will assign a color to each\n",
          # point in the mesh [x_min, x_max]x[y_min, y_max].\n",
          x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
          y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                               np.arange(y_min, y_max, self.h))
          Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
          # Put the result into a color plot\n",
          Z = Z.reshape(xx.shape)

          if type_classifier == 'Logistic':
             plt.pcolormesh(xx, yy, Z, shading='auto', cmap=self.cmap_light)
             # Plot also the training points\n",
             plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap=self.cmap_bold,
                   edgecolor='k', s=20)
             plt.title("Phase vs Amplitude (Logistic Regression)", fontsize=12)
             plt.axis('tight')
             plt.xlabel('Amplitude', fontsize=12)
             plt.ylabel('Phase', fontsize=12)
             plt.ylim([-4,4])
             plt.xlim([0,250])
             plt.show()
             for i in range(number_of_experiment):
                 path = type_classifier+"_"+"Experiments"
                 image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
                 image_folder = os.path.join(path,image_directory)

                 if not os.path.isdir("./"+image_folder):
                    os.system("mkdir "+image_folder) 
             i=1
             while os.path.exists("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
                 i+=1

             #plt.savefig("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png",bbox_inches='tight')
             plt.clf()

             cm = confusion_matrix(self.y_test,y_pred)
             self.plot_confusion_matrix(cm,self.c)
  


      def KMeans(self):
      ## Training
          k_means = KMeans(n_clusters=2)
          k_means.fit(self.X_train)
          y_pred = k_means.predict(self.X_test)            #labels or c

          # Plot the decision boundary. For that, we will assign a color to each\n",
          # point in the mesh [x_min, x_max]x[y_min, y_max].\n",
          x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
          y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                               np.arange(y_min, y_max, self.h))
          Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
          # Put the result into a color plot\n",
          Z = Z.reshape(xx.shape)

          if type_classifier == 'KMeans':
             plt.pcolormesh(xx, yy, Z, shading='auto', cmap=self.cmap_light)
             # Plot also the training points\n",
             plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap=self.cmap_bold,
                   edgecolor='k', s=20)
             plt.title("Phase vs Amplitude (k-means)", fontsize=12)
             plt.axis('tight')
             plt.xlabel('Amplitude', fontsize=12)
             plt.ylabel('Phase', fontsize=12)
             plt.xlim([0,250])
             plt.ylim([-4,4])
             plt.show()
             for i in range(number_of_experiment):
                 path = type_classifier+"_"+"Experiments"
                 image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
                 image_folder = os.path.join(path,image_directory)

                 if not os.path.isdir("./"+image_folder):
                    os.system("mkdir "+image_folder) 
             i=1
             while os.path.exists("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
                 i+=1

             #plt.savefig("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png",bbox_inches='tight')
             plt.clf()
        
             cm = confusion_matrix(self.y_test,y_pred)
             self.plot_confusion_matrix(cm,self.c)
             plt.clf()



      def Gaussian_Mixture_Model(self):
      ## Training
          gmm = mixture.GaussianMixture(n_components=2)
          gmm.fit(self.X_train)
          y_pred = gmm.predict(self.X_test)
          y_pred_old = gmm.predict(self.X_test)
          y_pred[y_pred_old==0] = 1
          y_pred[y_pred_old==1] = 0
          y_pred[y_pred_old==2] = 2

          ## Drawing Decision Boundaries
          # Plot the decision boundary. For that, we will assign a color to each\n",
          # point in the mesh [x_min, x_max]x[y_min, y_max].\n",
          x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
          y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                               np.arange(y_min, y_max, self.h))
          Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
          Z_old = np.copy(Z)
          Z[Z_old==0] = 1
          Z[Z_old==1] = 0
          Z[Z_old==2] = 2
          # Put the result into a color plot\n",
          Z = Z.reshape(xx.shape)

          if type_classifier == 'GMM':
             plt.pcolormesh(xx, yy, Z, shading='auto', cmap=self.cmap_light)
             # Plot also the training points\n",
             plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap=self.cmap_bold,
                   edgecolor='k', s=20)
             plt.title("Phase vs Amplitude (Gaussian Mixture Model)", fontsize=12)
             plt.axis('tight')
             plt.xlabel('Amplitude', fontsize=12)
             plt.ylabel('Phase', fontsize=12)
             plt.xlim([0,250])
             plt.ylim([-4,4])
             plt.show()
             for i in range(number_of_experiment):
                 path = type_classifier+"_"+"Experiments"
                 image_directory = type_classifier+"_"+"Images" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
                 image_folder = os.path.join(path,image_directory)

                 if not os.path.isdir("./"+image_folder):
                    os.system("mkdir "+image_folder) 
             i=1
             while os.path.exists("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png"):
                 i+=1

             #plt.savefig("./"+image_folder+"/" +type_classifier +f" Decision_bound {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)+".png",bbox_inches='tight')
             plt.clf()

             cm = confusion_matrix(self.y_test,y_pred)
             self.plot_confusion_matrix(cm,self.c)
             plt.clf()


        
      def pickle_data(self): 
          for i in range(number_of_experiment):
              name = type_classifier+"_"+"Experiments"
              dir_name = type_classifier+"_"+"Experiments" +"-" +"SNR_"+str(SNR)+"_"  +"RFI_" +str(SNR_high)
              new_folder = os.path.join(name,dir_name)

              if not os.path.isdir("./"+new_folder):
                 os.system("mkdir "+new_folder) 

          i=1
          while os.path.exists("./"+new_folder+"/" +type_classifier +f" Experiment {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high)):
              i+=1
          output = open("./"+new_folder+"/" +type_classifier +f" Experiment {i} - SNR= " +str(SNR) +f", RFI=" +str(SNR_high), 'wb')
          pickle.dump(self.cm,output)   
          output.close() 
          #print('cm2=', self.cm) 



def main():
    argv = sys.argv[1:]

    try:
       opts, args = getopt.getopt(argv, 'n:s:r:a:f:c:e:t:', ['num_sources', 'SNR', 'SNR_high', 'pareto_number', 'fov', 'num_channels_to_corrupt', 'number_of_experiment', 'type_classifier='])
       print('Options:', opts)

    except getopt.GetoptError:
       # Print something useful
       print ("-n : Number of sources created") 
       print ("-s : SNR value for noise")
       print ("-r : SNR value for RFI")
       print ("-a : Pareto number is 2")
       print ("-f : field of view is 10")
       print ("-c : Number of channels to be corrupted eg. 5")
       print ("-e : Number of experiment to perform")
       print ("--type_classifier : Type of classifier eg. Naive, Logistic, KMeans, GMM")
       sys.exit(2)

    for opt, arg in opts:
       if opt in ("-n"):
          num_sources = int(arg)

       elif opt in ("-s"):
          SNR = int(arg)
       elif opt in ("-r"):
          SNR_high = int(arg)
       elif opt in ("-a"):
          pareto_number = int(arg)
       elif opt in ("-f"):
          fov = int(arg)
       elif opt in ("-c"):
          num_channels_to_corrupt = int(arg)
       elif opt in ("-e"):
          number_of_experiment = int(arg)
       elif opt in ("-t","--type_classifier"):
          type_classifier = arg
     
    print ("Number of sources = ", num_sources) 
    print ("Value of thermal noise SNR = ", SNR)
    print ("Value of RFI SNR = ", SNR_high)
    print ("Field of view(fov) = ", fov)
    print ("Pareto number(a) = ", pareto_number)
    print ("Number of channels to corrupt = ", num_channels_to_corrupt)
    print ("Number of experiment to perform = ", number_of_experiment)
    print ("Type of classifier = ", type_classifier)
    return num_sources, SNR,SNR_high,fov,pareto_number,num_channels_to_corrupt,number_of_experiment,type_classifier


          
  

if __name__ =="__main__":
   num_sources, SNR,SNR_high,fov,pareto_number,num_channels_to_corrupt,number_of_experiment,type_classifier=main()
   s = sim()
   s.latitude()
   s.declination()
   s.lam()
   s.hour_angle_range()
   s.antenna_positions()
   s.num_antennas()
   s.num_baselines()
   s.uv_tracks()
   s.plot_uv_coverage()
   s.single_baseline()
   s.plot_uv_track()

   s.generate_flux()
   s.generate_pos()
   s.create_point_sources()
   s.baseLength_to_XYZ()

   s.create_vis()

   s.mask()
   s.det_power_of_signal()
   s.adding_noise()

   s.adding_RFI()

   s.create_colour_maps()
   s.classification_parameters()

   s.scattered_plot()
   s.Naive_Bayes_Classifier()
   #s.Logistic_Regression()
   #s.KMeans()
   #s.Gaussian_Mixture_Model()
   s.pickle_data()

  
 
#python Pipeline.py -n 100 -s 5 -r -17 -a 2 -f 10 -c 5 -e 2 --type_classifier Naive
#python Restart.py -n 100 -s 5 -r -17 -a 2 -f 10 -c 5 -e 2 --type_classifier Naive






