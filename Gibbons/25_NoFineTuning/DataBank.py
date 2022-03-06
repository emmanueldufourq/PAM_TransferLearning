import os.path
from os import path
import random
import numpy as np
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import randint
from tensorflow.keras.utils import to_categorical
import pickle

class DataBank:
    ''' Manage the data. Allow for data to be sampled in a 
    reproducible manner.
    '''
    
    def __init__(self, species_folder, test_fraction, augmentation_amount, type):
        
        self.species_folder = species_folder
        self.seed = self.initialise_seed()
        self.test_fraction = test_fraction
        self.label_encoder = LabelEncoder()
        self.augmentation_amount = augmentation_amount
        self.type = type

    def initialise_seed(self):
        ''' Initialise a seed to a random integer.
        '''
        return (random.randint(2, 1147483648))

    def set_seed(self, seed):
        self.seed = seed
                
    def get_seed(self):
        ''' Get the seed value
        '''
        return self.seed
    
    def generate_new_seed(self):
        ''' Creat a new seed
        '''
        self.seed = self.initialise_seed()
        
    def __load_picked_data(self):
        '''
        Load all of the spectrograms from a pickle file
        
        '''

        # Check which type of pre-processed input to use
        # duplicated spectrograms, different hop values
        # or powered values.
        if self.type == 'hop':
            X_data_name = 'X-hop.pkl'
            Y_data_name = 'Y-hop.pkl'

        if self.type == 'dup':
            X_data_name = 'X-dup.pkl'
            Y_data_name = 'Y-dup.pkl'

        if self.type == 'pow':
            X_data_name = 'X-pow.pkl'
            Y_data_name = 'Y-pow.pkl'

        if self.type == 'pow2':
            X_data_name = 'X-pow2.pkl'
            Y_data_name = 'Y-pow2.pkl'
        
        if self.type == 'test':
            X_data_name = 'X-test.pkl'
            Y_data_name = 'Y-test.pkl'
        
        if path.exists(os.path.join(self.species_folder, 'Saved_Data', X_data_name)) == False:
            print ('Pickled Data X does not exist.\nCreate it first.')
            raise Exception('Pickled Data X does not exist.')
            
        if path.exists(os.path.join(self.species_folder, 'Saved_Data', Y_data_name)) == False:
            print ('Pickled Data Y does not exist.')
            raise Exception('Pickled Data Y does not exist.\nCreate it first.')
        
        infile = open(os.path.join(self.species_folder, 'Saved_Data', X_data_name),'rb')
        X = pickle.load(infile)
        infile.close()
        
        infile = open(os.path.join(self.species_folder, 'Saved_Data',Y_data_name),'rb')
        Y = pickle.load(infile)
        infile.close()

        return X, Y
        
    def __randomly_sample_data(self, amount_to_sample, X, Y, positive_class_label):
        ''' Randomly sample a given amount of examples from the positive class
        without replacement. All the examples from the negative class remain 
        constant. E.g. randomly get 20 examples of Cape Robin-Chat calls
        will return X and Y values which contain all the original noise
        and only 20 examples of Cape Robin-Chats.
        '''
        
        # Get all indices of the examples which match the positive class
        species_indices = np.where(Y == positive_class_label)[0]
        
        # Get all indices of the examples which match the negative class
        non_species_indices = np.where(Y != positive_class_label)[0]
        
        # If the number of elements to randomly sample is greater than
        # the amount of data, then sample with replacement.
        if amount_to_sample > len(species_indices):
            
            # Randomly select (with replacement)
            randomly_selected_idx = list(np.random.choice(list(species_indices), amount_to_sample, replace=True))
        else:
            
            # Randomly select (without replacement)
            randomly_selected_idx = random.sample(list(species_indices), amount_to_sample)
        
        # Get the spectrograms of the species of interest
        X_augmented, Y_augmented = self.__augment_with_time_shift(X[randomly_selected_idx], positive_class_label)

        # Get the spectrograms and labels of background noise
        X_background = X[list(non_species_indices)]
        Y_background = Y[list(non_species_indices)]

        X = np.concatenate((X_augmented, X_background))
        Y = np.concatenate((Y_augmented, Y_background))
        
        # Shuffling needed
        X, Y = self.__shuffle_data(X, Y)
        
        # Return the randomly selected examples
        return X, Y
        
    def __shuffle_data(self, X, Y):
        ''' Shuffle the X, Y paired data
        '''
        X, Y = shuffle(X, Y, random_state=self.seed)
        
        return X,Y

    def __create_traintest_split(self, X, Y):
        ''' Split the data into training and testing
        '''
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
            test_size=self.test_fraction, random_state=self.seed, 
            shuffle=True)
        
        return X_train, X_test, Y_train, Y_test


    def __augment_with_time_shift(self, X_data, positive_class_label):
        ''' Augment the spectrograms by shifting them to some random
        time steps to the right.
        '''
        augmented_spectrograms = []
        augmented_labels = []

        # Iterate over all the spectrograms for the species of
        # interest.
        for X in X_data:

            # Create a number of augmentated spectrograms
            for i in range (0, self.augmentation_amount):

                # Randomly select amount to shift by
                random_shift = randint(1, X.shape[1]-1)

                # Time shift
                shifted_spectrogram = np.roll(X, random_shift, axis=1)

                # Append the augmented segments
                augmented_spectrograms.append(shifted_spectrogram)

                # Append the class labels
                augmented_labels.append(positive_class_label)

        # Return the augmented spectograms and labels
        return np.asarray(augmented_spectrograms), np.asarray(augmented_labels)
      
    def get_data(self, amount_to_sample, positive_class_label, call_order):
        ''' Randomly sample a pre-defined number of X,Y pairs
        while keeping the examples from the background noise
        constant and randomly sampling from the positive class.
        '''
        # Set the seed
        random.seed(self.get_seed())
        
        # Read in the X and Y data
        X, Y = self.__load_picked_data()

        unique, counts = np.unique(Y, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        print('Original Data distribution:',original_distribution)

        # Randomly sample from the data
        X_train, Y_train = self.__randomly_sample_data(amount_to_sample, 
            X, Y, positive_class_label)

        print ('Unique Y_train',np.unique(Y_train))
        print(len(np.unique(Y_train)))

        unique, counts = np.unique(Y_train, return_counts=True)
        train_distribution = dict(zip(unique, counts))
        print('Sampled Data distribution:',train_distribution)

        # Transform the Y labels into one-hot encoded labels
        for index, call_type in enumerate(call_order):
            Y_train = np.where(Y_train == call_type, index, Y_train)

        Y_train = to_categorical(Y_train, num_classes = len(np.unique(Y_train)))

        del X, Y

        # Return the spectrograms, labels and the seed used
        # The full test set is returned.
        # The sampled training set is returned (i.e. the training set
        # will only contained `amount_to_sample` number of examples of
        # the species of interest)
        return X_train, Y_train, original_distribution, train_distribution, self.seed
