import os
import sys
import pandas as pd
import numpy as np
import rioxarray
import utility_functions


proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

#Load the image corresponding to case study area (Italian Sea until 50 meters depth)
case_study = rioxarray.open_rasterio(proj_dir+'\\data\\case_study\\italy_bathymetry_50_00_4km.tif')
case_study_squeezed=case_study.squeeze(dim='band')
case_study_image=case_study_squeezed.to_numpy()



#Save width and height of the image
height,width = case_study_image.shape



#Prepare the image in order to have a grid
image_array=np.reshape(case_study_image, -1)


#Load the table of the input variables, latitude, londitude and the value of the pixel index
input_path = proj_dir+'\\data\\raw\\input_table.csv'
data = pd.read_csv(input_path)


#Scale the input variables
new_data=data.drop(labels=['lat','lon','pixel_indices'],axis=1)
scaler = utility_functions.get_scaler(new_data)
features=new_data.to_numpy()
input_data = scaler.transform(features)

#Set the number of neighbours of the grid
num_neighbours=4
#Set the radius to compute the mask for the neighbours
if num_neighbours == 4:
    radius=1.2
else:
    radius = 1.5
nan_values=-33333
neighbours=utility_functions.compute_image_neighbours(num_neighbours, radius, image_array, width, height, nan_values)


#set the correlation value (None for without correlation case)
correlation=[None]

#compute the list of nearest pixel only for the case study area based on correlation values
for element in correlation:
    lista_index_total=utility_functions.compute_index_list_for_adjMatrix(None,neighbours, data, input_data,num_neighbours,'pixel_indices')
#Compute the indices of the adj matrix for the graph
 
    source=data.pixel_indices.to_numpy()
    target=data.pixel_indices.to_numpy()
    source = np.empty((1,),dtype=int)
    target = np.empty((1,),dtype=int)

    new_indices=[]

    for indices in lista_index_total:
        new_indices = []
        for j in range(num_neighbours+1):
            if indices[j] != None:
                new_indices.append(indices[j])
        if len(new_indices) > 1:
            for i in range(len(new_indices)):

                source=np.append(source,new_indices[i])
                target=np.append(target,indices[int(num_neighbours/2)])
    source = np.delete(source, 0, 0)
    target=np.delete(target, 0, 0)

    df_for_adj1 = pd.DataFrame({'source':source ,'target':target}).dropna()

    df = pd.crosstab(df_for_adj1.source, df_for_adj1.target)

    df[df > 1] = 1
    adj_numpy=df.to_numpy()
    adj_numpy
#make adj undirected
    for i in range(0,adj_numpy.shape[0]):
        for j in range (0,adj_numpy.shape[1]):
            if adj_numpy[i,j] != adj_numpy[j,i]:
                if adj_numpy[i,j]== 0:
                    adj_numpy[i,j]=1
                if adj_numpy[j,i]== 0:
                    adj_numpy[j,i]=1

    edge_indices_source=[]
    edge_indices_target=[]
    for i in range(adj_numpy.shape[0]):
        for j in range(adj_numpy.shape[1]):
            if adj_numpy[i, j] == 1 and i!=j:
                edge_indices_source.append(i)
                edge_indices_target.append(j)

#Save the adj matrix
    output_path = proj_dir+"\\data\\adj_matrices\\adj_matrix_"+str(num_neighbours)+"neig_"+str(element)+"_correlation.csv"
    df=pd.DataFrame({'source':edge_indices_source,'target':edge_indices_target})
    df.to_csv(output_path,index=False)
