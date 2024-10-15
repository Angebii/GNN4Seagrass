import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

#UTILITY FUNCTIONS

#To scale the input variables
def get_scaler(node_features):
    scaler = StandardScaler()
    features = node_features.to_numpy()
    scaler.fit(features)
    return scaler

#To compute a grid of pixels to use in order to compute the graph
#It requires:
# - the number of neighbours pixels to use to create a grid (num_neighbours - int)
# - the radius to compute the mask for the grid (radius - float)
# - the image as a numpy array (image_array - numpy)
# - the width of the image (width - int)
# - the height of the image (height - int)
# - the values of the raster image used for nan values (nan_values - int)
def compute_image_neighbours(num_neighbours, radius, image_array, width, height, nan_values):
    neighbours = np.empty((1, num_neighbours + 1), dtype=int)
    for index in range(image_array.size):
        if image_array[index] > nan_values:
            row, column = index // width, index % width
            radius = radius
            r = int(radius)
            x=np.arange(max(column - r, 0), min(column + r+1,width))
            y=np.arange(max(row - r, 0), min(row + r+1,height))
            X,Y=np.meshgrid(x,y)
            R=np.sqrt(((X-column)**2 + (Y - row)**2))
            mask=R<radius
            neighbors=Y[mask]*width + X[mask]

            if len(neighbors) < num_neighbours+1:
                #it means that the pixel has a number of neighbours less than num_neighbours (e.g. coastal pixels)
                #so the list of num_neighbours is filled with -1
                for i in range(0,num_neighbours+1-len(neighbors)):
                    neighbors=np.append(neighbors,-1)

            for i in range(0,len(neighbors)):
                if neighbors[i] > image_array.size:
                    neighbors[i] = -1
                if image_array[neighbors[i]] < nan_values:
                    # it means that the value in the image corresponding to pixel index neighbors[i] is out of the case study
                    # so the list of num_neighbours is filled with -1
                    neighbors[i]=-1
            neighbours=np.append(neighbours,[neighbors],axis=0)
    neighbours=np.delete(neighbours, 0, 0)
    return neighbours



#To compute a grid of pixels only for the case study with different level of correlation to compute the graph
def compute_index_list_for_adjMatrix(correlation,neighbours, data, input_data,num_neighbours,column_name):
    lista_index_total=np.empty((1, num_neighbours+1), dtype=int)
    for i in range(len(neighbours)):
        lista_index=[]
        for j in range(0,num_neighbours+1):
            if len(data.index[data[column_name] == neighbours[i][j]]) != 0:
                index1=data.index[data[column_name]== neighbours[i][j]][0]
                for n in range(0,num_neighbours+1):
                    if len(lista_index) == num_neighbours+1:
                        lista_index=[]
                    t1=torch.tensor(input_data[index1])
                    if neighbours[i][n] != -1:
                        if len(data.index[data[column_name] == neighbours[i][n]]) != 0:
                            if correlation == None:
                                lista_index.append(neighbours[i][n])
                            else:
                                index2 = data.index[data[column_name] == neighbours[i][n]][0]
                                t2=torch.tensor(input_data[index2])
                                cos = torch.nn.CosineSimilarity(dim=0)
                                output = cos(t1, t2)
                                if output>=correlation and correlation>=0:
                                    lista_index.append(neighbours[i][n])
                                elif output<0 and correlation < 0:
                                    lista_index.append(neighbours[i][n])
                                else:
                                    lista_index.append(None)
                        else:
                            lista_index.append(None)
                    else:
                        lista_index.append(None)
        arr=np.array(lista_index)
        if len(arr) != 0:
            lista_index_total = np.append(lista_index_total, [arr], axis=0)
    lista_index_total=np.delete(lista_index_total, 0, 0)
    return lista_index_total
