import os
import numpy as np
import pickle
import pandas as pd
from scipy.io import loadmat
import random


def preprocess_USCHAD():
    N_USER = 14
    N_ACTIVITY = 12
    N_TRIAL = 5

    root_folder='data/USC-HAD'
    window_size=128
    features=3 

    all_user = dict()

    for usr_idx in range(1, N_USER+1):
        dat = dict()
        for activity in range(1, N_ACTIVITY+1):            
            all_readings = dict()
            for trial in range(1, N_TRIAL+1):
                filename = f'{root_folder}/Subject{usr_idx:d}/a{activity:d}t{trial:d}.mat'
                data = loadmat(filename)
                all_readings[trial] = data['sensor_readings']
            dat[activity] = all_readings
        all_user[usr_idx] = dat

    windows_users=[]
    activity_users=[]

    for i in range(1, N_USER+1):
        windows=[]
        activity=[]
        for j in range(1, N_ACTIVITY+1):
            for k in range(1, N_TRIAL+1):
                n_wind=len(all_user[i][j][k])//window_size
                for x in range(n_wind-1):
                    begin=x*window_size
                    end=(x+1)*window_size
                    begin_ov=int((x+0.5)*window_size)
                    end_ov=int((x+1.5)*window_size)

                    windows.append(all_user[i][j][k][begin:end])
                    activity.append(j)
                    windows.append(all_user[i][j][k][begin_ov:end_ov])
                    activity.append(j)

        windows_users.append(windows)
        activity_users.append(activity)

    N_ACT=4 #number of activities considered

    windows_users_subset=[]
    activity_users_subset=[]

    act_len_USCHAD=np.zeros((N_USER, N_ACT))

    for i in range(len(windows_users)):
        windows=[]
        activity=[]
        for j in range(len(windows_users[i])):
            if(activity_users[i][j]==1):
                act_len_USCHAD[i][0]+=1
                windows.append(windows_users[i][j])
                activity.append([1, 0, 0, 0])
            if(activity_users[i][j]==6):
                act_len_USCHAD[i][1]+=1
                windows.append(windows_users[i][j])
                activity.append([0, 1, 0, 0])
            if(activity_users[i][j]==7):
                act_len_USCHAD[i][2]+=1
                windows.append(windows_users[i][j])
                activity.append([0, 0, 1, 0])
            if(activity_users[i][j]==9):
                act_len_USCHAD[i][3]+=1
                windows.append(windows_users[i][j])
                activity.append([0, 0, 0, 1])
        windows_users_subset.append(windows)
        activity_users_subset.append(activity)



    #NORMALIZE by 6g

    norm_const=6

    for i in range(len(windows_users_subset)):
        for j in range(len(windows_users_subset[i])):
            windows_users_subset[i][j]=windows_users_subset[i][j]/(norm_const)

    #DOWNSAMPLING TO 50 Hz (original samplinh frequency is 100 Hz)

    wind_size=64

    for i in range(len(windows_users_subset)):
        for j in range(len(windows_users_subset[i])):
            new_wind=np.zeros((wind_size, features*2))
            for k in range(len(windows_users_subset[i][j])):
                if(k%2==0):
                    new_wind[int(k/2)]=windows_users_subset[i][j][k]
            windows_users_subset[i][j]=np.array(new_wind)

    #SHUFFLING 

    for i in range(len(windows_users_subset)):
        temp = list(zip(windows_users_subset[i], activity_users_subset[i]))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        windows_users_subset[i], activity_users_subset[i] = list(res1), list(res2)

    return windows_users_subset, activity_users_subset

###############################################################
###############################################################
###############################################################


DATA_PATH = 'RealWorld/'
N_REAL=15
N_ACT=4

BATCH_SIZE = 64
COL_NAMES = ['attr_x',	'attr_y',	'attr_z']
DATA_MAP = {'walking': [1,0,0,0],
            'running': [0,1,0,0],
            'jumping': [0,0,1,0],
            'standing': [0,0,0,1]
            }
RANGE = 19.6

def extract_data_arr(path_to_csv):
    df = pd.read_csv(path_to_csv)
    columns_to_extract = COL_NAMES
    data_array = df[columns_to_extract].values/RANGE
    num_submatrices = len(data_array) // BATCH_SIZE
    data_array_reshaped = data_array[:BATCH_SIZE * num_submatrices].reshape(num_submatrices, BATCH_SIZE, -1)
    return data_array_reshaped



def preprocess_RealWorld():

    files=["acc_walking_waist.csv", "acc_walking_2_waist.csv", "acc_running_waist.csv", "acc_jumping_waist.csv", "acc_standing_waist.csv", "acc_standing_2_waist.csv"]
    data_dict = {}
    for sub in os.listdir(DATA_PATH):
        # print(sub)
        data_dict[sub] = {}
        for act_ in os.listdir(os.path.join(DATA_PATH,sub, 'data')):
            if act_ in files:
                # print(act_)
                act = act_.split('_')[1]
                one_hot_act = tuple(DATA_MAP[act])
                data_dict[sub][one_hot_act] = extract_data_arr(os.path.join(DATA_PATH,sub, 'data', act_))

    with open(DATA_PATH + '../' + 'data_dict.pickle', "wb") as pickle_file:
        pickle.dump(data_dict, pickle_file)

    one_hot_act=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    one_hot_dic=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

    objects = []
    with (open("data_dict.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    act_len_users=np.zeros((N_REAL, N_ACT))

    for i in range(1, N_REAL+1):
        act_len_users[i-1][0]=int(len(objects[0]['proband'+str(i)][(1, 0, 0, 0)]))
        act_len_users[i-1][1]=int(len(objects[0]['proband'+str(i)][(0, 1, 0, 0)]))
        act_len_users[i-1][2]=int(len(objects[0]['proband'+str(i)][(0, 0, 1, 0)]))
        act_len_users[i-1][3]=int(len(objects[0]['proband'+str(i)][(0, 0, 0, 1)]))

    wind_real=[]
    wind_real_act=[]
    act_len=np.zeros((N_REAL, N_ACT))

    for i in range(N_REAL):
        wind_user=[]
        wind_user_act=[]
        for j in range(N_ACT):
            wind_us=[]
            wind_act=[]
            act_len[i][j]=int(act_len_users[i][j]-20)
            for k in range(10, int(act_len_users[i][j]-10)):
                wind_us.append(objects[0]['proband'+str(i+1)][one_hot_dic[j]][k])
                wind_act.append(one_hot_act[j])

            wind_user.append(wind_us)
            wind_user_act.append(wind_act)

        wind_real.append(wind_user)
        wind_real_act.append(wind_user_act)

    for i in range(N_REAL):
        wind_real[i]=np.concatenate(wind_real[i])
        wind_real_act[i]=np.concatenate(wind_real_act[i])

    for i in range(N_REAL):

        temp = list(zip(wind_real[i], wind_real_act[i]))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        wind_real[i], wind_real_act[i] = list(res1), list(res2)

    return wind_real, wind_real_act



