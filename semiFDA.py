import model_functions as f

import numpy as np
import tensorflow as tf
import random


def load_data(data_server, data_clients):
    clients_X=np.load(data_clients, allow_pickle=True)['windows_users']
    clients_y=np.load(data_clients, allow_pickle=True)['activity_users']

    for i in range(len(clients_X)):
        clients_X[i]=np.array(clients_X[i])

    #load data for server

    server_X=np.load(data_server, allow_pickle=True)['windows_users']
    server_y=np.load(data_server, allow_pickle=True)['activity_users']

    server_X=np.concatenate(server_X)
    server_X=np.array(server_X[:, :, :3])

    server_y=np.concatenate(server_y)
    server_y=np.array(server_y)

    return clients_X, clients_y, server_X, server_y


def construct_sets(server_X, server_y, clients_X, clients_y, num_classes, N_TOT, N_USER):
    #shuffle of server's data

    temp = list(zip(server_X, server_y))
    random.shuffle(temp) 
    res1, res2 = zip(*temp)
    server_X, server_y = list(res1), list(res2)

    server_X=np.array(server_X)
    server_y=np.array(server_y)

    #count data in each class on the server
    classes_server=np.zeros(num_classes)

    for i in range(len(server_y)):
        c=np.argmax(server_y[i])
        classes_server[int(c)]+=1


    #select randomly a subset of users to use as clients (if needed)

    list_user=[]
    for i in range(N_TOT):
        list_user.append(i)

    random.shuffle(list_user)
    list_user=np.array(list_user)

    sub_clients_X=[]
    sub_clients_y=[]
    for i in range(N_USER):
        sub_clients_X.append(clients_X[list_user[i]])
        sub_clients_y.append(clients_y[list_user[i]])

    # sub_clients_X=np.array(sub_clients_X)
    # sub_clients_y=np.array(sub_clients_y)

    #split data of each client in training (80%) and testing (20%) data

    wind_clients=[]
    for i in range(N_USER):
        wind_clients.append(len(sub_clients_X[i]))

    wind_clients=np.array(wind_clients)

    train_clients_X=[]
    train_clients_y=[]
    test_clients_X=[]
    test_clients_y=[]

    for i in range(N_USER):
        begin=wind_clients[i]//100*80
        train_clients_X.append(sub_clients_X[i][:begin, :, :3])
        train_clients_y.append(sub_clients_y[i][:begin])
        test_clients_X.append(sub_clients_X[i][begin:, :, :3])
        test_clients_y.append(sub_clients_y[i][begin:])


    #count data in each class on the clients

    classes_test_users=np.zeros((N_USER, num_classes))

    for i in range(N_USER):
        for j in range(len(test_clients_y[i])):
            c=np.argmax(test_clients_y[i][j])
            classes_test_users[i][int(c)]+=1


    #data used by a client in each round
    wind_round_clients=wind_clients//100*8

    return server_X, server_y, train_clients_X, train_clients_y, test_clients_X, test_clients_y, wind_round_clients



def compute_acc_full(model, test_clients_X, test_clients_y):
    acc_temp=[]
    for j in range(len(test_clients_X)):        
        predClassTest=model.predict(test_clients_X[j])
        acc_temp.append(f.my_validation(test_clients_y[j], predClassTest))

    return acc_temp

def compute_acc_SSFL(model_enc, model_class, test_clients_X, test_clients_y):
    acc_temp=[]
    for j in range(len(test_clients_X)):
        
        features_test=model_enc(test_clients_X[j][:, :, :3])
        predClassTest=model_class.predict(features_test)
        acc_temp.append(f.my_validation(test_clients_y[j], predClassTest))

    return acc_temp


def clients_training(client_models_SSFL, client_models_SemiFDA, client_models_fedAvg, wind_round_clients, round, train_clients_X, train_clients_y, cov_matrix_server, lr, epochs_clients):
    cl=0
    cl_list=[]
    for s, fSemiFDA, fAvg  in zip(client_models_SSFL, client_models_SemiFDA, client_models_fedAvg):
        
        begin=wind_round_clients[cl]*round
        end=wind_round_clients[cl]*(round+1)
        tr_X_client= np.array(train_clients_X[cl][begin:end, :, :3])
        tr_y_client= np.array(train_clients_y[cl][begin:end])

        cl_list.append(tr_X_client)        

        n = tr_X_client.shape[0]  # Number of times to replicate the matrix

        cov_matrix_train = np.tile(cov_matrix_server, (n, 1, 1))

    
        s.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse') 
        s.fit(x=tr_X_client, y=tr_X_client, epochs=epochs_clients, batch_size=64, verbose=0)

        fSemiFDA.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=f.my_loss) 
        fSemiFDA.fit(x=tr_X_client, y=cov_matrix_train, epochs=epochs_clients, batch_size=64, verbose=0)

        fAvg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy') 
        fAvg.fit(x=tr_X_client, y=tr_y_client, epochs=epochs_clients, batch_size=64, verbose=0)

        cl+=1

    return client_models_SSFL, client_models_SemiFDA, client_models_fedAvg





def run_experiments(data_server, data_clients, run_numbers, save_title, comm_rounds, epochs_server, epochs_clients, n_user, n_tot):

    
    callback_full=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_ae=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lr=1e-3
    num_classes=4

    metrics=['accuracy']

    window_size=64
    features=3 

    COMM_ROUNDS=comm_rounds
    bn=16

    #load data for clients

    clients_X, clients_y, server_X, server_y = load_data(data_server, data_clients)


    start_server_list=[]
    start_clients_list=[]
    start_clients_SemiFDA_list=[]
    start_clients_SSFL_list=[]
    SSFL_retraining_list=[]
    SSFL_aggregation_list=[]
    SemiFDA_list=[]
    fedAvg_list=[]

    #starting runs

    for run in range(run_numbers):

        print("Run number: ", run+1)

        #setting the seed 

        random.seed(run)
        tf.random.set_seed(run)
        np.random.seed(run)

        server_X, server_y, train_clients_X, train_clients_y, test_clients_X, test_clients_y, wind_round_clients= construct_sets(server_X, server_y, clients_X, clients_y, num_classes, n_tot, n_user)


        print("Starting pretraining models on Server's data")

        full_model_SemiFDA=f.create_full(window_size, features, num_classes)
        full_model_SemiFDA.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy')
        full_model_SemiFDA.fit(x=server_X[:, :, :3], y=np.array(server_y), validation_split=0.2, callbacks=[callback_full], epochs=epochs_server, batch_size=64, verbose=0)  

        fedAvg_model=f.create_full(window_size, features, num_classes)
        fedAvg_model.set_weights(full_model_SemiFDA.get_weights())  

        # print("Shape test cients and labels", test_clients_X.shape, test_clients_y.shape)

        #compute accuracy of Centralized model on clients' test sets
        start_clients_list.append(compute_acc_full(fedAvg_model, test_clients_X, test_clients_y))


        #split full classifier in encoder and classification head

        encoder_SemiFDA=f.create_encoder_cov(window_size, features)
        for i in range(1,11):
            encoder_SemiFDA.layers[i].set_weights(full_model_SemiFDA.layers[i].get_weights())
        encoder_SemiFDA.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=f.my_loss)

        classifier_SemiFDA=f.create_classification_head(num_classes)
        for i in range(0, 5):
            classifier_SemiFDA.layers[i].set_weights(full_model_SemiFDA.layers[i+11].get_weights())
        classifier_SemiFDA.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=metrics)

        #compute initial accuracy of SemiFDA on clients' test sets
        start_clients_SemiFDA_list.append(compute_acc_full(full_model_SemiFDA, test_clients_X, test_clients_y))

        #compute COV matrix of server's data
        extractor_SemiFDA = tf.keras.Model(inputs=encoder_SemiFDA.inputs, outputs=encoder_SemiFDA.layers[10].output)
            
        features_SemiFDA=extractor_SemiFDA(server_X[:, :, :3]) 
        cov_matrix_server=np.cov(features_SemiFDA, rowvar=False)  

        #compute initial accuracy of SemiFDA on server data
        features_server=extractor_SemiFDA(server_X[:, :, :3])
        predClass_source=classifier_SemiFDA.predict(features_server)
        acc_server_SemiFDA=f.my_validation(server_y, predClass_source)

        start_server_list.append(acc_server_SemiFDA)


        #pretrain AE for Zhao's solution

        ae_SSFL=f.create_ae(window_size, features)
        ae_SSFL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
        ae_SSFL.fit(x=server_X[:, :, :3], y=server_X[:, :, :3], validation_split=0.2, callbacks=[callback_ae], epochs=epochs_server, batch_size=64, verbose=0)

        encoder_SSFL = tf.keras.Model(inputs=ae_SSFL.inputs, outputs=ae_SSFL.layers[10].output)
        features_SSFL=encoder_SSFL(server_X[:, :, :3])

        classifier_SSFL=f.create_classification_head(num_classes)
        classifier_SSFL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=metrics)
        classifier_SSFL.fit(x=features_SSFL, y=np.array(server_y), validation_split=0.2, callbacks=[callback_full], epochs=epochs_server, batch_size=64, verbose=0)

        print("Pretraining of Centralized model ended")

        #create client models 

        client_models_SSFL=f.create_client_models_ssfl(ae_SSFL, n_user, lr)
        client_models_SemiFDA=f.create_client_models_SemiFDA(encoder_SemiFDA, n_user, lr)
        client_models_fedAvg=f.create_client_models(fedAvg_model, n_user, lr)


        #compute initial accuracy of Zhao's solution on clients' test sets
        start_clients_SSFL_list.append(compute_acc_SSFL(encoder_SSFL, classifier_SSFL, test_clients_X, test_clients_y))


        features_server=encoder_SSFL(server_X[:, :, :3])
        predClass_source=classifier_SSFL.predict(features_server)
        acc_server_SSFL=f.my_validation(server_y, predClass_source)


        #create lists to save accuracies during communication rounds
        acc_aggregation_SSFL=[] 
        acc_retraining_SSFL=[]
        acc_SemiFDA=[]
        acc_fedAvg=[]

        print("Starting Communication Rounds")

        for round in range(COMM_ROUNDS):
            print("Communication Round number:", round+1)

            #training on clients' data
            print("Training on clients...")

            client_models_SSFL, client_models_SemiFDA, client_models_fedAvg = clients_training(client_models_SSFL, client_models_SemiFDA, client_models_fedAvg, wind_round_clients, round, train_clients_X, train_clients_y, cov_matrix_server, lr, epochs_clients)

            print("Aggregation phase...")

            #aggregation of clients models in fedAvg
            client_weights_fedAvg=[client_models_fedAvg[i].get_weights() for i in range(0, n_user)]
            averaged_weights_fedAvg = [np.average(w, axis=0) for w in zip(*client_weights_fedAvg)]
            fedAvg_model.set_weights(averaged_weights_fedAvg)

            for z in range(0,n_user):
                client_models_fedAvg[z].set_weights(averaged_weights_fedAvg)

            #evaluate accuracy of fedAvg after the aggregation 
            acc_fedAvg.append(compute_acc_full(fedAvg_model, test_clients_X, test_clients_y))


            #aggregation of clients models in Zhao's solution 
            client_weights_SSFL=[client_models_SSFL[i].get_weights() for i in range(0, n_user)]
            averaged_weights_SSFL = [np.average(w, axis=0) for w in zip(*client_weights_SSFL)]
            ae_SSFL.set_weights(averaged_weights_SSFL)
            encoder_SSFL = tf.keras.Model(inputs=ae_SSFL.inputs, outputs=ae_SSFL.layers[10].output)

            for z in range(0,n_user):
                client_models_SSFL[z].set_weights(averaged_weights_SSFL)

            
            #aggregation of clients models in SemiFDA
            client_weights_SemiFDA=[client_models_SemiFDA[i].get_weights() for i in range(0, n_user)]
            averaged_weights_SemiFDA = [np.average(w, axis=0) for w in zip(*client_weights_SemiFDA)]
            encoder_SemiFDA.set_weights(averaged_weights_SemiFDA)

            for z in range(0,n_user):
                client_models_SemiFDA[z].set_weights(averaged_weights_SemiFDA)
            
            for i in range(1, 11):
                full_model_SemiFDA.layers[i].set_weights(encoder_SemiFDA.layers[i].get_weights())

        
            #evaluate accuracy of Zhao's solution and SemiFDA after the aggregation 
            acc_aggregation_SSFL.append(compute_acc_SSFL(encoder_SSFL, classifier_SSFL, test_clients_X, test_clients_y))
            acc_SemiFDA.append(compute_acc_full(full_model_SemiFDA, test_clients_X, test_clients_y))

            print("Retraining of classification head for Zhao's solution...")
            #fine-the classification head for Zhao's solution
            features_SSFL=encoder_SSFL(server_X[:, :, :3])
            classifier_SSFL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=metrics)
            classifier_SSFL.fit(x=features_SSFL, y=np.array(server_y), validation_split=0.2, callbacks=[callback_full], epochs=epochs_server, batch_size=64, verbose=0)


            #evaluate accuracy of Zhao's solution after the fine-tuning of the classification head
            acc_retraining_SSFL.append(compute_acc_SSFL(encoder_SSFL, classifier_SSFL, test_clients_X, test_clients_y))


        #append each run results

        SSFL_aggregation_list.append(acc_aggregation_SSFL)
        SSFL_retraining_list.append(acc_retraining_SSFL)
        SemiFDA_list.append(acc_SemiFDA)
        fedAvg_list.append(acc_fedAvg)

        #save run results in a .npz file

        print("Saving data...")

        np.savez(save_title + str(run), start_clients_list=start_clients_list, fedAvg_list=fedAvg_list, acc_aggregation_SSFL=acc_aggregation_SSFL, SSFL_aggregation_list=SSFL_aggregation_list, SSFL_retraining_list=SSFL_retraining_list, SemiFDA_list=SemiFDA_list, start_clients_SSFL_list=start_clients_SSFL_list, start_clients_SemiFDA_list=start_clients_SemiFDA_list, acc_server_SSFL=acc_server_SSFL, start_server_list=start_server_list)


    np.savez(save_title + "_tot", start_clients_list=start_clients_list, fedAvg_list=fedAvg_list, acc_aggregation_SSFL=acc_aggregation_SSFL, SSFL_aggregation_list=SSFL_aggregation_list, SSFL_retraining_list=SSFL_retraining_list, SemiFDA_list=SemiFDA_list, start_clients_SSFL_list=start_clients_SSFL_list, start_clients_SemiFDA_list=start_clients_SemiFDA_list, acc_server_SSFL=acc_server_SSFL, start_server_list=start_server_list)
