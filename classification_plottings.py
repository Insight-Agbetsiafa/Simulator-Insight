import matplotlib
import itertools 
import matplotlib.cm as cm
import pickle

import sklearn
from sklearn.linear_model import LogisticRegression as logis
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

'''
MACHINE LEARNING/CLASSIFICATION
'''
class sim():
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
          #plt.show()
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
          y_pred[y_pred_old==0] = 2
          y_pred[y_pred_old==1] = 1
          y_pred[y_pred_old==2] = 0

          ## Drawing Decision Boundaries
          # Plot the decision boundary. For that, we will assign a color to each\n",
          # point in the mesh [x_min, x_max]x[y_min, y_max].\n",
          x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
          y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
          xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                               np.arange(y_min, y_max, self.h))
          Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
          Z_old = np.copy(Z)
          Z[Z_old==0] = 2
          Z[Z_old==1] = 1
          Z[Z_old==2] = 0
          # Put the result into a color plot\n",
          Z = Z.reshape(xx.shape)

          if type_classifier == 'GMM':
             plt.pcolormesh(xx, yy, Z, shading='auto', cmap=self.cmap_light)
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



      '''
      Function to save data to a specific directory according to the classifiers names and
      the corresponding thermal noise and RFI value
      '''        
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


    

if __name__ =="__main__":
   s = sim()
   s.create_colour_maps()
   s.classification_parameters()
   s.scattered_plot()
   s.Naive_Bayes_Classifier()
   #s.Logistic_Regression()
   #s.KMeans()
   #s.Gaussian_Mixture_Model()
   #s.pickle_data()
   
