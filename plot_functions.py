import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_results(path, comm_rounds):

    SSFL_aggregation_list=np.load(path + ".npz")['SSFL_aggregation_list']
    SSFL_retraining_list=np.load(path + ".npz")['SSFL_retraining_list']
    SemiFDA_list=np.load(path + ".npz")['SemiFDA_list']
    fedAvg_list=np.load(path + ".npz")['fedAvg_list']

    start_clients_list=np.load(path + ".npz")['start_clients_list']
    start_server_list=np.load(path + ".npz")['start_server_list']


    start_clients=np.mean(start_clients_list, axis=1)
    acc_aggregation_SSFL=np.mean(SSFL_aggregation_list, axis=2)
    acc_retraining_SSFL=np.mean(SSFL_retraining_list, axis=2)
    acc_SemiFDA=np.mean(SemiFDA_list, axis=2)
    acc_fedAvg=np.mean(fedAvg_list, axis=2)

    start_clients=np.mean(start_clients, axis=0)
    start_server=np.mean(start_server_list, axis=0)
    acc_aggregation_SSFL=np.mean(acc_aggregation_SSFL, axis=0)
    acc_retraining_SSFL=np.mean(acc_retraining_SSFL, axis=0)
    acc_SemiFDA=np.mean(acc_SemiFDA, axis=0)
    acc_fedAvg=np.mean(acc_fedAvg, axis=0)

    print("Initial Accuracy on Server data: ", start_server)

    print("Initial Accuracy on Clients test data", start_clients)
  
    # print("\n\nAccuracy after first aggregation SSFL", acc_aggregation_SSFL[0] )
    # print("Accuracy after first fine-tuning of classification head SSFL", acc_retraining_SSFL[0] )
    
    # print("\nFinal accuracy after aggregation SSFL", acc_aggregation_SSFL[comm_rounds-1] )
    print("Final accuracy after fine-tuning of classification head AE-SSFL", acc_retraining_SSFL[comm_rounds-1] )

    print("\nFinal accuracy SemiFDA", acc_SemiFDA[comm_rounds-1] )

    # print("\n\nAccuracy after first aggregation fedAvg", acc_fedAvg[0] )
    print("\nFinal accuracy fedAvg", acc_fedAvg[comm_rounds-1] )


def plot_results(path, title, a, b, comm_rounds, bands):

    #To check accuracy before server retraining for Zhao's solution
    # SSFL_aggregation_list=np.load(path + ".npz")['SSFL_aggregation_list']
    SSFL_retraining_list=np.load(path + ".npz")['SSFL_retraining_list']
    SemiFDA_list=np.load(path + ".npz")['SemiFDA_list']
    fedAvg_list=np.load(path + ".npz")['fedAvg_list']
    # accStart_SSFL=np.load(path + ".npz")['accStart_SSFL']

    start_clients_list=np.load(path + ".npz")['start_clients_list']


    # accStart_SSFL=np.mean(accStart_SSFL)
    start_clients=np.mean(start_clients_list, axis=1)
    # acc_aggregation_SSFL=np.mean(SSFL_aggregation_list, axis=2)
    acc_retraining_SSFL=np.mean(SSFL_retraining_list, axis=2)
    acc_SemiFDA=np.mean(SemiFDA_list, axis=2)
    acc_fedAvg=np.mean(fedAvg_list, axis=2)

    #To check accuracy before server retraining for Zhao's solution
    # std_ssfl_aggregation=3*np.std(acc_aggregation_SSFL, axis=0)/np.sqrt(len(acc_aggregation_SSFL))
    std_ssfl_retraining=3*np.std(acc_retraining_SSFL, axis=0)/np.sqrt(len(acc_retraining_SSFL))
    std_SemiFDA=3*np.std(acc_SemiFDA, axis=0)/np.sqrt(len(acc_SemiFDA))
    std_fedAvg=3*np.std(acc_fedAvg, axis=0)/np.sqrt(len(acc_fedAvg))
    std_start=3*np.std(start_clients, axis=0)/np.sqrt((len(start_clients)))

    start_clients=np.mean(start_clients, axis=0)
    # acc_aggregation_SSFL=np.mean(acc_aggregation_SSFL, axis=0)
    acc_retraining_SSFL=np.mean(acc_retraining_SSFL, axis=0)
    acc_SemiFDA=np.mean(acc_SemiFDA, axis=0)
    acc_fedAvg=np.mean(acc_fedAvg, axis=0)

    #To check accuracy before server retraining for Zhao's solution
    # pos_std_aggregation_ssfl=acc_aggregation_SSFL+std_ssfl_aggregation
    # neg_std_aggregation_ssfl=acc_aggregation_SSFL-std_ssfl_aggregation

    pos_std_retraining_ssfl=acc_retraining_SSFL+std_ssfl_retraining
    neg_std_retraining_ssfl=acc_retraining_SSFL-std_ssfl_retraining

    pos_std_SemiFDA=acc_SemiFDA+std_SemiFDA
    neg_std_SemiFDA=acc_SemiFDA-std_SemiFDA

    pos_std_fedAvg=acc_fedAvg+std_fedAvg
    neg_std_fedAvg=acc_fedAvg-std_fedAvg

    pos_std_start=start_clients+std_start
    neg_std_start=start_clients-std_start


    #To check accuracy before server retraining for Zhao's solution

    # acc_ssfl=[]
    # pos_std_ssfl=[]
    # neg_std_ssfl=[]


    # for i in range(comm_rounds):
    #     acc_ssfl.append(acc_aggregation_SSFL[i])
    #     acc_ssfl.append(acc_retraining_SSFL[i])
    #     pos_std_ssfl.append(pos_std_aggregation_ssfl[i])
    #     pos_std_ssfl.append(pos_std_retraining_ssfl[i])
    #     neg_std_ssfl.append(neg_std_aggregation_ssfl[i])
    #     neg_std_ssfl.append(neg_std_retraining_ssfl[i])

    # acc_ssfl=np.insert(acc_ssfl, 0, start_clients)
    # pos_std_ssfl=np.insert(pos_std_ssfl, 0, pos_std_start)
    # neg_std_ssfl=np.insert(neg_std_ssfl, 0, neg_std_start)

    acc_ssfl=np.insert(acc_retraining_SSFL, 0, start_clients)
    pos_std_ssfl=np.insert(pos_std_retraining_ssfl, 0, pos_std_start)
    neg_std_ssfl=np.insert(neg_std_retraining_ssfl, 0, neg_std_start)


    acc_fedAvg=np.insert(acc_fedAvg, 0, start_clients)
    pos_fedAvg=np.insert(pos_std_fedAvg, 0, pos_std_start)
    neg_fedAvg=np.insert(neg_std_fedAvg, 0, neg_std_start)
    
    acc_SemiFDA=np.insert(acc_SemiFDA, 0, start_clients)
    pos_SemiFDA=np.insert(pos_std_SemiFDA, 0, pos_std_start)
    neg_SemiFDA=np.insert(neg_std_SemiFDA, 0, neg_std_start)


    x_full=np.zeros(comm_rounds+1)
    for i in range(comm_rounds+1):
        x_full[i]=i

    #To check accuracy before server retraining for Zhao's solution

    # x_half=np.zeros(comm_rounds*2 +1)

    # for i in range(comm_rounds*2 +1):
    #     j=i/2
    #     x_half[i]=j


    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.ylim(a, b)

    plt.plot(x_full, acc_ssfl, color='red', label="AE-SSFL of Zhao")
    plt.plot(acc_SemiFDA, color='blue', label="SemiFDA")
    plt.plot(acc_fedAvg, color='black', ls='--', label="FedAvg (supervised)")


    if(bands):
        plt.plot(x_full, pos_std_ssfl, color='red', ls='--')
        plt.plot(x_full, neg_std_ssfl, color='red', ls='--')
        plt.plot(pos_SemiFDA, color='blue', ls='--')
        plt.plot(neg_SemiFDA, color='blue', ls='--')
        plt.plot(pos_fedAvg, color='grey', ls='--')
        plt.plot(neg_fedAvg, color='grey', ls='--')

        plt.fill_between(x_full, pos_std_ssfl, neg_std_ssfl, color='red', alpha=0.2)
        plt.fill_between(x_full, pos_SemiFDA, neg_SemiFDA, color='blue', alpha=0.2)
        plt.fill_between(x_full, pos_fedAvg, neg_fedAvg, color='grey', alpha=0.2)
        # plt.fill_between(x_full, acc_perc75_fedAVG, acc_perc25_fedAVG, color='red', alpha=0.2)


    for i in range(1, comm_rounds+1):
        plt.vlines(i, ymin=a, ymax=b, color='grey', ls='-')

    plt.hlines(start_clients, xmin=0, xmax=comm_rounds, color="black", ls='--', label="Centralized model")
    plt.xticks([i for i in range(0, comm_rounds+1)])
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.grid(axis='y')

    plt.legend()