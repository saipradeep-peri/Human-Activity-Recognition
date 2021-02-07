#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


## Cell which containes all the parameters
folderPath = 'PAMAP2_Dataset/Protocol'
filePath = 'subject101.dat'


# In[6]:


person1_data = pd.read_table(folderPath+'/'+filePath)


# In[7]:


person1_data.head()


# In[9]:


person1_data_numpy = person1_data.values
nrows, _ = person1_data_numpy.shape
ncols = 54

# convert the string of data for each row into array
person1_data_matrix = np.zeros((nrows, ncols))

# person1_data_list = list(list())
for i, row in enumerate(person1_data_numpy):
    row_list = row[0].split()
    row_array = np.asarray(row_list)
    row_array_floats = row_array.astype(np.float)
    person1_data_matrix[i, :] = row_array_floats


# In[11]:


person1_data_matrix.shape


# In[14]:


person1_data_matrix[0]


# In[15]:


activity_ID = person1_data_matrix[:, 1]
activity_ID


# In[17]:


activity_ID


# In[2]:


## Cell which containes all the parameters
folderPath = '/Users/nikunjgoel/Study/dataMining_2160/PAMAP2_Dataset/Protocol/'
outputFolderpath = '/Users/nikunjgoel/Study/dataMining_2160/PAMAP2_Dataset/'


# In[3]:


NUM_FILES_TO_READ = 8


# In[39]:


all_subjects_data_matrix = None
for i in range(1, NUM_FILES_TO_READ+1):
    data_file_name = folderPath + 'subject10{}.dat'.format(i)
    if all_subjects_data_matrix is None:
        all_subjects_data_matrix = read_data(data_file_name,i)
    else:
        all_subjects_data_matrix = np.append(all_subjects_data_matrix, read_data(data_file_name,i), axis=0)
    print("Files read: {}".format(i))
    print("Current data matrix size: {}".format(all_subjects_data_matrix.shape))

print("Shuffling data matrix and writing to file...")
# np.random.shuffle(all_subjects_data_matrix)
np.savetxt(outputFolderpath + 'cleanData.csv', all_subjects_data_matrix, fmt='%.5f', delimiter=",")
print("All done dawg")


# In[37]:


def read_data(data_file_name,iValue):
    """
        Read data from file_name, store in DataLoader
    """
    person1_data = pd.read_table(data_file_name)
    person1_data_numpy = person1_data.values
    nrows, _ = person1_data_numpy.shape
    ncols = 54

    # convert the string of data for each row into array
    person1_data_matrix = np.zeros((nrows, ncols))

    # person1_data_list = list(list())
    for i, row in enumerate(person1_data_numpy):
        row_list = row[0].split()
        row_array = np.asarray(row_list)
        row_array_floats = row_array.astype(np.float)
        person1_data_matrix[i, :] = row_array_floats

    # discard data that includes activityID = 0
    activity_ID = person1_data_matrix[:, 1]
    good_data_count = 0
    for i in range(nrows):
        if activity_ID[i] != 0:
            good_data_count += 1

    person1_data_matrix_fixed = np.zeros((good_data_count, ncols))
    count = 0
    for i in range(nrows):
        if activity_ID[i] != 0:
            person1_data_matrix_fixed[count, :] = person1_data_matrix[i, :]
            count += 1

    prev_heart_rate = np.nan
    # Fill in heart rate values with previous time-stamp values
    for i in range(np.alen(person1_data_matrix_fixed)):
        if not np.isnan(person1_data_matrix_fixed[i, 2]):
            prev_heart_rate = person1_data_matrix_fixed[i, 2]
            continue
        if np.isnan(person1_data_matrix_fixed[i, 2]) and not np.isnan(prev_heart_rate):
            person1_data_matrix_fixed[i, 2] = prev_heart_rate

    # Remove all rows with Nan
    person1_data_matrix_fixed = person1_data_matrix_fixed[~np.any(np.isnan(person1_data_matrix_fixed), axis=1)]
    person1_data_matrix_fixed = np.append(person1_data_matrix_fixed,np.full_like(np.zeros((person1_data_matrix_fixed.shape[0],1)),fill_value = iValue),1)

    return person1_data_matrix_fixed


# In[39]:


all_subjects_data_matrix = None
for i in range(1, NUM_FILES_TO_READ+1):
    data_file_name = folderPath + 'subject10{}.dat'.format(i)
    if all_subjects_data_matrix is None:
        all_subjects_data_matrix = read_data(data_file_name,i)
    else:
        all_subjects_data_matrix = np.append(all_subjects_data_matrix, read_data(data_file_name,i), axis=0)
    print("Files read: {}".format(i))
    print("Current data matrix size: {}".format(all_subjects_data_matrix.shape))

print("Shuffling data matrix and writing to file...")
# np.random.shuffle(all_subjects_data_matrix)
np.savetxt(outputFolderpath + 'cleanData.csv', all_subjects_data_matrix, fmt='%.5f', delimiter=",")
print("All done dawg")


# In[ ]:




