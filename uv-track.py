import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


'''
Creating uv-tracks
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
      print ("\n")
     

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
      Plots the uv-coverage of an array layout
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
          plt.show()
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
          plt.show()
          plt.clf()
             

     

if __name__ =="__main__":
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
   
   
   
