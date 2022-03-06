from xml.dom import minidom
import math
import pandas as pd
import glob, os
import librosa.display
import librosa
import numpy as np
from scipy import signal
from tensorflow.keras import backend, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
import gc


from yattag import Doc, indent
import ntpath

class Prediction:
    
    def __init__(self, species_folder, lowpass_cutoff, 
                 downsample_rate, nyquist_rate, segment_duration, 
                 positive_class, negative_class,
                 augmentation_amount_positive_class,
                 augmentation_amount_negative_class,
                 n_fft, hop_length, n_mels, f_min, f_max, weights_name):
        self.species_folder = species_folder
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.augmentation_amount_positive_class = augmentation_amount_positive_class
        self.augmentation_amount_negative_class = augmentation_amount_negative_class
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.audio_path = self.species_folder + '/Audio/'
        self.annotations_path = self.species_folder + '/Annotations/'
        self.saved_data_path = self.species_folder + '/Saved_Data/'
        self.testing_files = self.species_folder + '/DataFiles/TestingFiles.txt'
        self.n_ftt = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.weights_name = weights_name
        
        
    def read_audio_file(self, file_name):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"
        
        '''
        # Get the path to the file
        audio_folder = os.path.join(file_name)
        
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)
        
        return audio_amps, audio_sample_rate
    
    def create_X_new(self, mono_data, time_to_extract, sampleRate,start_index, end_index, verbose):
        '''
        Create X input data to apply a model to an audio file.
        '''

        X_frequences = []

        sampleRate = sampleRate
        duration = end_index - start_index -time_to_extract+1
        if verbose:
            # Print spme info
            print ('-----------------------')
            print ('start (seconds)', start_index)
            print ('end (seconds)', end_index)
            print ('duration (seconds)', (duration))
            print()
        counter = 0

        end_index = start_index + time_to_extract
        # Iterate over each chunk to extract the frequencies
        for i in range (0, duration):

            if verbose:
                print ('Index:', counter)
                print ('Chunk start time (sec):', start_index)
                print ('Chunk end time (sec):',end_index)

            # Extract the frequencies from the mono file
            extracted = mono_data[int(start_index *sampleRate) : int(end_index * sampleRate)]

            X_frequences.append(extracted)

            start_index = start_index + 1
            end_index = end_index + 1
            counter = counter + 1

        X_frequences = np.array(X_frequences)
        print (X_frequences.shape)
        if verbose:
            print ()

        return X_frequences

    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    def group_consecutives(self, vals, step=1):
        """Return list of consecutive lists of numbers from vals (number list)."""
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        '''
        Downsample an audio file to a given new sample rate.
        amplitudes:
        original_sr:
        new_sample_rate:
        
        '''
        return librosa.resample(amplitudes, 
                                original_sr, 
                                new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate

    def convert_single_to_image(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_ftt,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled

    
        # 3 different input
        return np.stack([S1,S1**3,S1**5], axis=2)

    def convert_all_to_image(self, segments):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)
    
    def add_keras_dim(self, spectrograms):
        spectrograms = np.reshape(spectrograms, 
                                  (spectrograms.shape[0],
                                   spectrograms.shape[1],
                                   spectrograms.shape[2],1))
        return spectrograms
    
    def load_model(self):
        
        INPUT_SHAPE = (128,216, 3)

        base_model = ResNet152V2(weights="imagenet",
                    input_shape=INPUT_SHAPE,
                    include_top=False)  

        inputs = Input(shape=INPUT_SHAPE)
        x = base_model(inputs, training=False)
        x = Flatten()(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs, outputs)

        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        print('loading: ', self.weights_name)
        model.load_weights("{}".format(self.weights_name))
        
        return model
    
    def group(self, L):
        L.sort()
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last: # Part of the group, bump the end
                last = n
            else: # Not part of the group, yield current group and start a new
                yield first, last
                first = last = n
        yield first, last # Yield the last group

    def dataframe_to_svl(self, dataframe, sample_rate, length_audio_file_frames):

        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        doc.asis('<!DOCTYPE sonic-visualiser>')

        with tag('sv'):
            with tag('data'):
                
                model_string = '<model id="1" name="" sampleRate="{}" start="0" end="{}" type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" subtype="box" minimum="0" maximum="{}" units="Hz" />'.format(sample_rate, 
                                                                            length_audio_file_frames,
                                                                            sample_rate/2)
                doc.asis(model_string)
                
                with tag('dataset', id='0', dimensions='2'):

                    # Read dataframe or other data structure and add the values here
                    # These are added as "point" elements, for example:
                    # '<point frame="15360" value="3136.87" duration="1724416" extent="2139.22" label="Cape Robin" />'
                    for index, row in dataframe.iterrows():
                        point  = '<point frame="{}" value="{}" duration="{}" extent="{}" label="{}" />'.format(int(int(row['start(sec)'])*sample_rate), int(row['low(freq)']),
                                                          int((int(row['end(sec)'])- int(row['start(sec)']))*sample_rate), 
                                                          "{:.2f}".format((int(row['high(freq)'])  - int(row['low(freq)']))),
                                                          row['label'])
                        
                        # add the point
                        doc.asis(point)
            with tag('display'):
                
                display_string = '<layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
                doc.asis(display_string)

        result = indent(
            doc.getvalue(),
            indentation = ' '*2,
            newline = '\r\n'
        )

        return result

    def predict_all_test_files(self, verbose):
        '''
        Create X and Y values which are inputs to a ML algorithm.
        Annotated files (.svl) are read and the corresponding audio file (.wav)
        is read. A low pass filter is applied, followed by downsampling. A 
        number of segments are extracted and augmented to create the final dataset.
        Annotated files (.svl) are created using SonicVisualiser and it is assumed
        that the "boxes area" layer was used to annotate the audio files.
        '''
        
        if verbose == True:
            print ('Annotations path:',self.annotations_path+"*.svl")
            print ('Audio path',self.audio_path+"*.wav")
        
        # Read all names of the training files
        testing_files = pd.read_csv(self.testing_files, header=None)
        
        model = self.load_model()
        
        df_data_file_name = []

        # Iterate over each annotation file
        for testing_file in testing_files.values:
            
            # Initialise dictionary to contain a list of all the seconds
            # which contain calls
            call_seconds = set()
            
            # Keep track of how many calls were found in the annotation files
            total_calls = 0
            
            file = self.annotations_path+'/'+testing_file[0]+'.svl'
            
            # Get the file name without paths and extensions
            file_name_no_extension = file[file.rfind('/')+1:file.find('.')]
            
            print ('Processing:',file_name_no_extension)
            
            df_data_file_name.append(file_name_no_extension)
            
            # Check if the .wav file exists before processing
            if self.audio_path+file_name_no_extension+'.WAV' in glob.glob(self.audio_path+"*.WAV"):

                # Read audio file
                audio_amps, original_sample_rate = self.read_audio_file(self.audio_path+file_name_no_extension+'.WAV')
                    
                print ('Applying filter')
                # Low pass filter
                filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)

                print ('Downsample')
                # Downsample
                filtered, filtered_sample_rate = self.downsample_file(filtered, 
                                                                      original_sample_rate, self.downsample_rate)

                print ('Creating segments')
                # Split the file into segments for prediction
                segments = self.create_X_new(filtered, self.segment_duration, 
                                        filtered_sample_rate,0, int(len(filtered)/filtered_sample_rate), False)

                print ('Converting to spectrogram')
                spectrograms = self.convert_all_to_image(segments)

                #spectrograms = self.add_keras_dim(spectrograms)

                print ('Predicting')
                # Predict on each segment
                model_prediction = model.predict(spectrograms)

                # Find all predictions which had a softmax value 
                # greater than some threshold
                values = model_prediction
                values = values[:,1]  >= 0.5
                values = values.astype(np.int)

                # Find all the seconds which contain positive predictions
                positive_seconds = np.where(values == 1)[0]

                # Group the predictions into consecutive chunks (to allow
                # for audio clips to be extracted)
                groups = self.group_consecutives(np.where(values == 1)[0])
                
                predictions = []
                for pred in groups:

                    if len(pred) >= 2:
                        #print (pred)
                        for predicted_second in pred:
                            # Update the set of all the predicted calls
                            predictions.append(predicted_second)
                
                predictions.sort()

                # Only process if there are consecutive groups
                if len(predictions) > 0:
                    predicted_groups = list(self.group(predictions))
                    
                    print ('Predicted')

                    # Create a dataframe to store each prediction
                    df_values = []
                    for pred_values in predicted_groups:
                        df_values.append([pred_values[0], pred_values[1]+self.segment_duration, 3000, 7000, 'predicted'])
                    df_preds = pd.DataFrame(df_values, columns=[['start(sec)','end(sec)','low(freq)','high(freq)','label']])

                    # Create a .svl outpupt file
                    xml = self.dataframe_to_svl(df_preds, original_sample_rate, len(audio_amps))

                    # Write the .svl file
                    text_file = open(self.species_folder+'/Model_Output/'+file_name_no_extension+".svl", "w")
                    n = text_file.write(xml)
                    text_file.close()

                # Clean up
                del spectrograms, filtered, audio_amps, groups
                gc.collect()
                gc.collect()

        return
