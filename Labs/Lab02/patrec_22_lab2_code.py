# %% [markdown]
# # Pattern Recognition 2022-23 1st Lab: Speech Recognition with Hidden Markov Models and Recurrent Neural Networks<br>
# 
# Ηλιόπουλος Γεώργιος: 03118815 <br>
# Σερλής Αναστάσιος Εμανουήλ - 03118125<br>
# <br>

# %% [markdown]
# ## Imports

# %%
#!pip install librosa
#!pip install pytorch

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [markdown]
# ## Libraries

# %%
import torch
import librosa
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
from word2number import w2n
import re
import IPython.display as ipd
import sounddevice as sd
import pandas as pd
from sklearn.decomposition import PCA

# %% [markdown]
# ## Step 2 - Data Parser

# %%
digits_dir = "./digits"
n_samples=133

def data_parser(dir):
    wavs = [None] * n_samples
    speakers = np.zeros(n_samples, dtype=int)
    digits = np.zeros(n_samples, dtype=int)

    for i, file in enumerate(os.listdir(dir)):
         fl=os.fsdecode(file) #read file name
         wavs[i], _ = librosa.load(os.path.join(dir, fl), sr=16000) #librosa load
         temp_list = re.split(r'(^[^\d]+)',fl.split('.')[0] )[1:] # split word and speaker
         digits[i] = w2n.word_to_num(temp_list[0]) #get digit 
         speakers[i] = int(temp_list[1]) #get speaker

    return wavs, speakers, digits

wavs,speakers,digits = data_parser(digits_dir)
for i in range(10): #print wav sample, speaker and digit for 10 first files
    print("wav_file: {}\t speaker = {}\t digit = {}\t samples = {}".format(i+1,speakers[i],digits[i],wavs[i][:3]))

print("We have {} wav files for {} digits from {} different speakers."\
      .format(len(wavs), np.unique(digits).shape[0], np.unique(speakers).shape[0]))
#check sound on speaker 1 and diff digits
sd.play(wavs[0], 16000) #speaker 1 - digit 8
sd.play(wavs[14],16000) #speaker 1 - digit 5

# %% [markdown]
# ## Step 3 - Extract MFCCs

# %%
sr=16000 #wanted parameters for mfcc
feats=13
win_length=0.025
step=0.01

mfccs=[]
delta=[]
delta2=[]

for i in range(n_samples): #compute mfccs, delta (1st and 2nd order)
    mfcc_temp = librosa.feature.mfcc(wavs[i], sr=sr, n_mfcc=feats, hop_length=int(win_length*sr), n_fft=int(step*sr))
    mfccs.append(mfcc_temp)
    delta.append(librosa.feature.delta(mfcc_temp))
    delta2.append(librosa.feature.delta(mfcc_temp, order=2))



# %%
print("MFCCS  shape is {} with shape {}.".format(type(mfccs[0]),mfccs[0].shape))
print("delta  shape is {} with shape {}.".format(type(delta[0]),delta[0].shape))
print("delta2 shape is {} with shape {}.".format(type(delta2[0]),delta2[0].shape))

# %% [markdown]
# ## Step 4 - Histograms of 1st and 2nd MFCCs

# %%
# AM1: 031 18 125 --> n1 = 5
# AM2: 031 18 815 --> n2 = 5 -> 6
n1=5
n2=6

# Extract 1st and 2nd mfcc of digit 5
mfcc1_n1 = [mfccs[i][0] for i in range(n_samples) if digits[i] == n1]
mfcc2_n1 = [mfccs[i][1] for i in range(n_samples) if digits[i] == n1]

# Extract 1st and 2nd mfcc of digit 6
mfcc1_n2 = [mfccs[i][0] for i in range(n_samples) if digits[i] == n2]
mfcc2_n2 = [mfccs[i][1] for i in range(n_samples) if digits[i] == n2]

#flattened lists
mfcc1_n1_fl=list(np.concatenate(mfcc1_n1).flat)
mfcc2_n1_fl=list(np.concatenate(mfcc2_n1).flat)
mfcc1_n2_fl=list(np.concatenate(mfcc1_n2).flat)
mfcc2_n2_fl=list(np.concatenate(mfcc2_n2).flat)


# Plot histograms
fig = plt.figure(figsize=(18,3))
ax1=fig.add_subplot(1, 4, 1)
plt.hist(mfcc1_n1_fl, bins=30)
ax1.title.set_text('1st MFCC of digit {}'.format(n1))
ax2=fig.add_subplot(1, 4, 2)
plt.hist(mfcc2_n1_fl, bins=30)
ax2.title.set_text('2st MFCC of digit {}'.format(n1))
ax3=fig.add_subplot(1, 4, 3)
plt.hist(mfcc1_n2_fl, bins=30)
ax3.title.set_text('1st MFCC of digit {}'.format(n2))
ax4=fig.add_subplot(1, 4, 4)
plt.hist(mfcc2_n2_fl, bins=30)
ax4.title.set_text('1st MFCC of digit {}'.format(n2))
plt.show()

# %%
#get 2 instances for each of the digits 5 and 6
speaker1=2
speaker2=14
index1_n1 = int(np.where((speakers == speaker1) & (digits == n1) )[0][0])
index2_n1 = int(np.where((speakers == speaker2) & (digits == n1) )[0][0])
index1_n2 = int(np.where((speakers == speaker1) & (digits == n2) )[0][0])
index2_n2 = int(np.where((speakers == speaker2) & (digits == n2) ) [0][0])

print('\033[1m' + "Samples selected for MFSC & MFCC correlation:" + '\033[0m')
print("Speaker={}\t digit={}\t index={}".format(speaker1,n1,index1_n1))
print("Speaker={}\t digit={}\t index={}".format(speaker1,n2,index1_n2))
print("Speaker={}\t digit={}\t index={}".format(speaker2,n1,index2_n1))
print("Speaker={}\t digit={}\t index={}".format(speaker2,n2,index2_n2))

#extract MFSCs for selected 4 samples
mfsc1_n1 = librosa.feature.melspectrogram(wavs[index1_n1], sr=sr, hop_length=int(win_length*sr), n_fft=int(step*sr), n_mels=feats)
mfsc1_n2 = librosa.feature.melspectrogram(wavs[index1_n2], sr=sr, hop_length=int(win_length*sr), n_fft=int(step*sr), n_mels=feats)
mfsc2_n1 = librosa.feature.melspectrogram(wavs[index2_n1], sr=sr, hop_length=int(win_length*sr), n_fft=int(step*sr), n_mels=feats)
mfsc2_n2 = librosa.feature.melspectrogram(wavs[index2_n2], sr=sr, hop_length=int(win_length*sr), n_fft=int(step*sr), n_mels=feats)


# %%
# plot correlation for MFSC for each instance
fig = plt.figure(figsize=(15,12))

#plt.title("HI\n")

ax1=fig.add_subplot(1, 4, 1)
ax1.title.set_text('Correlation of MFSC for\nSpeaker {} and Digit {}'.format(speaker1, n1))
plt.imshow((pd.DataFrame.from_records(mfsc1_n1.T)).corr())

ax3=fig.add_subplot(1, 4, 2)
ax3.title.set_text('Correlation of MFSC for\nSpeaker {} and Digit {}'.format(speaker1, n2))
plt.imshow((pd.DataFrame.from_records(mfsc1_n2.T)).corr())

ax5=fig.add_subplot(1, 4, 3)
ax5.title.set_text('Correlation of MFSC for\nSpeaker {} and Digit {}'.format(speaker2, n1))
plt.imshow((pd.DataFrame.from_records(mfsc2_n1.T)).corr())

ax7=fig.add_subplot(1, 4, 4)
ax7.title.set_text('Correlation of MFSC for\nSpeaker {} and Digit {}'.format(speaker2, n2))
plt.imshow((pd.DataFrame.from_records(mfsc2_n2.T)).corr())


plt.show()

# %%
# plot correlation for MFCC for each instance
fig = plt.figure(figsize=(15,12))

ax1=fig.add_subplot(1, 4, 1)
ax1.title.set_text('Correlation of MFCC for\nSpeaker {} and Digit {}'.format(speaker1, n1))
plt.imshow((pd.DataFrame.from_records(mfccs[index1_n1].T)).corr())

ax2=fig.add_subplot(1, 4, 2)
ax2.title.set_text('Correlation of MFCC for\nSpeaker {} and Digit {}'.format(speaker1, n2))
plt.imshow((pd.DataFrame.from_records(mfccs[index1_n2].T)).corr())


ax3=fig.add_subplot(1, 4, 3)
ax3.title.set_text('Correlation of MFCC for\nSpeaker {} and Digit {}'.format(speaker2, n1))
plt.imshow((pd.DataFrame.from_records(mfccs[index2_n1].T)).corr())

ax4=fig.add_subplot(1, 4, 4)
ax4.title.set_text('Correlation of MFCC for\nSpeaker {} and Digit {}'.format(speaker2, n2))
plt.imshow((pd.DataFrame.from_records(mfccs[index2_n2].T)).corr())

plt.show()

# %% [markdown]
# ## Step 5 - Feature Extraction

# %%
#get global feature extraction
# 133 rows which correspond to the total samples
# 6*13 columns which correspond to the mean values and variance for each one of the 13 different mfcc features

features = np.zeros((n_samples, 6*13))
for i in range(n_samples):
    features[i, :13] = np.mean(mfccs[i], axis=1)
    features[i, 13:26] = np.mean(delta[i], axis=1)
    features[i, 26:39] = np.mean(delta2[i], axis=1)
    features[i, 39:52] = np.std(mfccs[i], axis=1)
    features[i, 52:65] = np.std(delta[i], axis=1)
    features[i, 65:  ] = np.std(delta2[i], axis=1)

colors = ['orange', 'black', 'orange', 'pink', 'gray', 'green', 'purple', 'brown', 'red'] #set different colors
numbers=[i for i in range(1,10)] #set digit values
symbols = ['.', 'o', 'v', '^', 'x', 'D', "X", 's', 'P']


def scatter(feats, y, dims=2, PCA=1):
    if dims == 2: #in case of 2 feats
        #plt.rcParams['figure.figsize'] = [10, 5]
        fig, ax = plt.subplots()
        feat0 = feats[:, 0]
        feat1 = feats[:, 1]
        for i in range(9):
             ax.scatter(feat0[y == i+1], feat1[y == i+1], c=colors[i],
                    label=numbers[i], s=50, alpha=1,marker = symbols[i])
             ax.set_xlabel('Feature 0')
             ax.set_ylabel('Feature 1')
             if(PCA==0):
                 ax.title.set_text('Scatter plot for 2 features before PCA')
             if(PCA==1):
                 ax.title.set_text('Scatter plot for 2 features after PCA')
            
    if dims == 3: #in case of 3 feats
        plt.rcParams['figure.figsize'] = [7, 7]
        fig = plt.figure()
        ax = plt.axes(projection ="3d")
        feat0 = feats[:, 0]
        feat1 = feats[:, 1]
        feat2 = feats[:, 2]
        for i in range(9):
            ax.scatter(feat0[y == i+1], feat1[y == i+1], feat2[y == i+1], c=colors[i],
                    label=numbers[i], s=50, alpha=1,marker = symbols[i])
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')
            ax.set_zlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()
    plt.show()

plt.rcParams['figure.figsize'] = [10, 5]    
np.random.shuffle(colors)
np.random.shuffle(symbols)
scatter(features,digits,2,PCA=0)

# %% [markdown]
# ## Step 6 - PCA

# %%
#set 2d and 3d PCAs

PCA_2D= PCA(n_components=2)
PCA_3D= PCA(n_components=3)

#reduce to two and three dims via PCA
feats_pca2d = PCA_2D.fit_transform(features)
feats_pca3d = PCA_3D.fit_transform(features)

# find varinace ratios
print("Variance ratio after 2D PCA is:",(PCA_2D.explained_variance_ratio_[0])*100,"%","and",(PCA_2D.explained_variance_ratio_[1])*100,"%")
print("Variance ratio after 3D PCA is:",(PCA_3D.explained_variance_ratio_[1])*100,"%","and",(PCA_3D.explained_variance_ratio_[1])*100,"%","and",(PCA_3D.explained_variance_ratio_[2])*100,"%")

# %%
#scatter plot for 2d-pca
#np.random.shuffle(colors)
#np.random.shuffle(symbols)
plt.rcParams['figure.figsize'] = [7, 7] 
scatter(feats_pca2d,digits,2,PCA=1)

# %%
#scatter plot for 3d-pca
#np.random.shuffle(colors)
#np.random.shuffle(symbols)
scatter(feats_pca3d,digits,3,PCA=1)

# %%
print(PCA_2D.explained_variance_ratio_)
print(np.sum(PCA_2D.explained_variance_ratio_))
print(PCA_3D.explained_variance_ratio_)
print(np.sum(PCA_3D.explained_variance_ratio_))

# %%
pcas = []
max_d = 15
var = []

for i in range(max_d):
    pca_iD = PCA(n_components = i)
    pca_iD.fit_transform(features)
    var.append(np.sum(pca_iD.explained_variance_ratio_)*100)

# %%
fig=plt.figure(figsize=(7,3))
plt.plot(range(max_d), var)
plt.ylabel("Toal Varience kept (%)")
plt.xlabel("Principal Component Number")
plt.annotate('2D PCA',xy=(2,79.18),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.2'),xytext=(0,85))
plt.annotate('3D PCA',xy=(3,86.627),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.2'),xytext=(4,70))
plt.grid()

# %% [markdown]
# ## Step 7 - Classification

# %%
# sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# %%
from scipy.stats import multivariate_normal as mn

# %%
#importing Custom Naive Bayes Classifier from 1st lab with a few changes
class CustomNBClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the Gaussian Naive Bayes"""

    def __init__(self):
        self.X_mean_ = None
        self.X_var_ = None
        self.prior = None
        self.classes = None
        self.n_classes = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        Calculates self.X_mean_ and self.X_var_ based on the mean 
        feature values in X for each class. Also, calculates self.prior
        that contains the prior probability of each class.
        
        self.X_mean_ becomes a numpy.ndarray of shape 
        (n_classes, n_features)
        
        self.X_var_ becomes a numpy.ndarray of shape 
        (n_classes, n_features)
        
        self.prior becomes a numpy.array of shape 
        (n_classes)
        
        fit always returns self.
        """

        # Initialize useful variables
        self.classes = np.unique(y)
        train_size, n_features = X.shape
        self.n_classes = len(self.classes)
        self.X_mean_ = np.zeros((self.n_classes, n_features))
        self.X_var_ = np.zeros((self.n_classes, n_features))

        # Compute mean and variance values for each class
        for count, value in enumerate(self.classes):
            idx_i = [i for i in range(train_size) if y[i] == value]
            X_i = np.take(X, idx_i, axis=0)
            self.X_mean_[count, :] = np.mean(X_i, axis=0, keepdims=True)
            self.X_var_[count, :] = np.var(X_i, axis=0, keepdims=True)
        
        # Compute prior probabilities for each class
        self.prior = np.zeros(self.n_classes)
        for i, value in enumerate(self.classes):
            self.prior[i] = np.count_nonzero(y == value) / train_size
        return self
        

    def predict(self, X):
        """
        Make predictions for X based on
        the highest posterior probability
        """

        # Compute likelihood
        like = np.zeros((self.n_classes, len(X)))
        for i in range(self.n_classes):
            like[i] = np.prod(1/(np.sqrt(2*np.pi*self.X_var_[i]+ 10**(-9))) * \
                      np.exp(-0.5*((X - self.X_mean_[i])**2 / (self.X_var_[i] + 10**(-9)))) \
                      , axis=1)

        return np.argmax(like.T * self.prior, axis=1) + 1
    
    def score(self, X, y): #returns accuracy
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        count = 0
        y_pred = self.predict(X)
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                count += 1
        acc = count / len(y)
        return acc

# %%
#splitting data in test and train sets
X_train, X_test, y_train, y_test = train_test_split(features, digits, test_size=0.3, random_state=42, stratify=digits)

#normalizing data
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# %%
def score_print(classifier):
    cl = classifier
    print('\033[1m' + cl.__class__.__name__ + '\033[0m')
    cl.fit(X_train, y_train)
    a = cl.score(X_test, y_test)
    print("Normalized Data: {},\taccuracy: {}".format("no", a))
    cl = classifier
    cl.fit(X_train_norm, y_train)
    a = cl.score(X_test_norm, y_test)
    print("Normalized Data: {},\taccuracy: {}".format("yes", a))
    print()

# %%
score_print(CustomNBClassifier())
score_print(GaussianNB())
score_print(KNeighborsClassifier(n_neighbors=3))
score_print(LogisticRegression())
score_print(SVC(kernel="linear", probability=True))

# %% [markdown]
# ## Step 8 - Pytorch

# %%
# pytorch
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch import optim

# %%
#generating our data
f = 40
T = 1/f
step = 0.001
X = np.zeros((1500, 10))
y = np.zeros((1500, 10))

for i in range(len(X)):
    A = np.random.rand() #getting a random amplitude (0,1)
    start = np.random.rand() * (T) #getting a random starting point
    time = np.linspace(start, start+step*10, num=10)
    X[i] = A*np.sin(2*np.pi*f*time)
    y[i] =  A*np.cos(2*np.pi*f*time)

# %%
# plotting some of the data
fig = plt.figure(figsize=(16,3))

plt.subplot(1,4,1)
plt.plot(range(10), X[1])
plt.title("sine no1")
plt.ylim([-0.65, 0.65])
plt.grid()

plt.subplot(1,4,2)
plt.plot(range(10), y[1], color ="purple")
plt.title("cosine no1")
plt.ylim([-0.65, 0.65])
plt.grid()

plt.subplot(1,4,3)
plt.plot(range(10), X[10])
plt.title("sine no10")
plt.ylim([-0.65, 0.65])
plt.grid()

plt.subplot(1,4,4)
plt.plot(range(10), y[10], color ="purple")
plt.title("cosine no10")
plt.ylim([-0.65, 0.65])
plt.grid()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# %%
# Defining a LSTM model
class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTMCell(1, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

# %%
# model parameters
model = LSTMNet().double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
#Training
train_losses = []
test_losses = []

#epochs = 1500
epochs = 1500
for i in range(epochs):
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    if i%100 == 0:
        print("Epoch:\t{}".format(i), end='\t')
        print('Train loss', loss.item(),end='\t')
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    with torch.no_grad():
        pred = model(X_test)
        loss = criterion(pred, y_test)
        if i%100 == 0:
            print('Test loss:', loss.item())
        test_losses.append(loss.item())

# %%
#plot LSTM losses
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
print(np.shape(train_losses))
plt.plot(np.arange(np.shape(train_losses)[0]),train_losses,label='Train Losses',color='b')
plt.plot(np.arange(np.shape(test_losses)[0]),test_losses,label='Test Losses',color='g')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('LSTM Losses')
plt.show()

# %%
#comparison to ground truth values
outputs_LSTM = model.forward(X_test[:4, :]).detach().numpy()

# plotting some of the data
fig = plt.figure(figsize=(16,3))

plt.subplot(1,4,1)
plt.plot(range(10), y_test[0])
plt.scatter(range(10),outputs_LSTM[0],color='red')
plt.title("LSTM Test No.1")

plt.subplot(1,4,2)
plt.plot(range(10), y_test[1])
plt.scatter(range(10),outputs_LSTM[1],color='red')
plt.title("LSTM Test No.2")

plt.subplot(1,4,3)
plt.plot(range(10), y_test[2])
plt.scatter(range(10),outputs_LSTM[2],color='red')
plt.title("LSTM Test No.3")

plt.subplot(1,4,4)
plt.plot(range(10), y_test[3])
plt.scatter(range(10),outputs_LSTM[3],color='red')
plt.title("LSTM Test No.4")

# %% [markdown]
# ## Step 9 - Free Spoken Digit Dataset (FSDD) - read, normalization, MFCCs and split

# %%
from glob import glob
import librosa as l
import os
from sklearn.preprocessing import StandardScaler
%pip install tqdm
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# %% [markdown]
# Parser function was given to us to read and extract features from FSDD.

# %%
def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

# %% [markdown]
# Importing the data and extracting mfccs.

# %%
data_dir = './recordings'
X_train, X_test, y_train, y_test, spk_train, spk_test = parser(data_dir)

# %% [markdown]
# Splitting into training and validation set (80% - 20%). Then we normalize the data.

# %%
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("If using all data to calculate normalization statistics")
scale_fn = make_scale_fn(X_train + X_dev + X_test)

print("If using X_train + X_dev to calculate normalization statistics")
scale_fn = make_scale_fn(X_train + X_dev)

print("If using X_train to calculate normalization statistics")
scale_fn = make_scale_fn(X_train)

X_train = scale_fn(X_train)
X_dev = scale_fn(X_dev)
X_test = scale_fn(X_test)

# %%
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# %% [markdown]
# what_is_it is a function to see info about our data.

# %%
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def what_is_it (data):
    y = namestr(data, globals())
    print('\033[1m' + y[0] + ':\033[0m')
    print("Type of data:\t\t {}".format(type(data)))
    print("Length of data:\t\t {}".format(len(data)))
    print()
    print("Each element of {} has:\ntype:\t\t\t {}".format(y[0], type(data[0])))
    if type(data[0]) == np.ndarray:
        print("example shape:\t\t {}".format(data[0].shape))
        print("shape of [0][0]:\t {}".format(data[0][0].shape))
    print()

# %%
what_is_it(X_train)
what_is_it(X_test)
what_is_it(X_dev)
what_is_it(y_train)
what_is_it(y_test)
what_is_it(y_dev)
what_is_it(spk_train)
what_is_it(spk_test)

# %%
print("Train size:\t\t{}".format(len(X_train)))
print("Validation size:\t{}".format(len(X_dev)))
print("Test size:\t\t{}".format(len(X_test)))

# %% [markdown]
# According to the above and by studying the ready fuctions given to us we underastand that each sample is a 2d NumPy array that each row represents a frame and each collumn represents a MFCC.

# %% [markdown]
# ## Step 10 - GMM-HMM

# %%
from pomegranate import *

# %% [markdown]
# Defining a list to keep the samples categorized by digit for the train set.

# %%
digit_pos_train = []

for digit in range(10):  
    digit_pos_train.append(digit)
    digit_pos_train[digit] = []
    for i in range(len(X_train)):
        if y_train[i] == digit:
            digit_pos_train[digit].append(i)

X_train_per_digit = []
for i in range(10):
    a = np.take(X_train, digit_pos_train[i], axis=0)
    X_train_per_digit.append(np.take(X_train, digit_pos_train[i], axis=0))

# %% [markdown]
# Defining a list to keep the samples categorized by digit for the validation set.

# %%
digit_pos_dev = []

for digit in range(10):  
    digit_pos_dev.append(digit)
    digit_pos_dev[digit] = []
    for i in range(len(X_dev)):
        if y_dev[i] == digit:
            digit_pos_dev[digit].append(i)

X_dev_per_digit = []
for i in range(10):
    a = np.take(X_dev, digit_pos_dev[i], axis=0)
    X_dev_per_digit.append(np.take(X_dev, digit_pos_dev[i], axis=0))

# %% [markdown]
# Defining the model nad the training function.

# %%
def train_GMM_HMM (X, n_states = 2, n_mixtures = 2, gmm = True, max_iterations = 30):

    # X is the data from a single digit (can be a numpy array)
    # n_states is the number of HMM states
    # n_mixtures is the number of Gaussians
    # gmm defines whether to use GMM or plain Gaussian

    dists = [] # list of probability distributions for the HMM states

    # GMM or plain Gaussian
    for i in range(n_states):
        if gmm and n_mixtures > 1:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, (np.vstack(X)).astype('float64'))
            # we use "np.vstack(X)" to stack all all frames in a single array
        else:
            a = MultivariateGaussianDistribution.from_samples((np.vstack(X)).astype('float64'))
        dists.append(a)

    # building the transition matrix
    trans_mat = np.zeros((n_states, n_states)) # transition matrix all zeros except the below "if"s
    for i in range(n_states):
        for j in range(n_states):
            if i == j or j == i+1:
                trans_mat[i, j] = 0.5 # setting aij = 0.5 if j = i+1 or i = j (transitions only between succesive states)

    starts = numpy.zeros(n_states) # your starting probability matrix
    starts[0] = 1

    ends = np.zeros(n_states) # your ending probability matrix
    ends[-1] = 1

    data = X # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
              # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])
    
    max_iter = max_iterations
    # Fit the model
    model.fit(X, max_iterations = max_iter)
    
    return model

# %% [markdown]
# ## Step 11 - Training the models

# %% [markdown]
# Making the models (10), one for each digit.

# %%
models = []
for i in range(10):
    models.append(train_GMM_HMM((X_train_per_digit[i]), n_states = 4, n_mixtures = 4))

# %% [markdown]
# ## Step 12 - Testing the models

# %% [markdown]
# Function that finds the predictions based on the calculation of the log likelihood for each recording of the validation set.

# %%
def predictions(models, X):
    
    preds = np.zeros(len(X), dtype='int') # predictions
    
    for i in range(len(X)):
        logs = np.zeros(10)
        
        for j in range(10):
            logp, _ = models[j].viterbi(X[i]) # Calculation of the log likelihood for each recording with viterbi algorithm
            logs[j] = logp # Saving the log likelihood for each model
        preds[i] = np.argmax(logs) # Selecting the max log likelihood
    
    return preds

# %%
preds = predictions(models, X_dev)
print("The predicttions for the first 10 samples are: {}\nwhile the real values are:{}".format(preds[0:10], y_dev[0:10]))

# %% [markdown]
# Function that returns the accuracy of a model.

# %%
def accuracy (predictions, labels):
    count = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            count = count + 1
    return count/len(labels)

# %%
acc = accuracy(predictions(models, X_dev), y_dev)
print("Validation set:")
print("The accuracy of the model is {:.3f}.".format(acc))
print("this model has 2 HMM states and 2 GMMs.")

# %%
acc = accuracy(predictions(models, X_test), y_test)
print("Test set:")
print("The accuracy of the model is {:.3f}.".format(acc))
print("this model has 2 HMM states and 2 GMMs.")

# %% [markdown]
# Getting experimntal with the parameters of the model to select the best combination.

# %%
from IPython.display import clear_output

# %%
states = [1, 2, 3 ,4]
GMMs= [1, 2, 3, 4, 5]

accuracies = []

for state in states:
    for gmm in GMMs:
        models = []
        print("...currently running model with {} HMM states and {} Gaussian distirbutions...".format(state, gmm))
        for i in range(10):
            models.append(train_GMM_HMM(X_train_per_digit[i], n_states = state, n_mixtures = gmm))
        acc = accuracy(predictions(models, X_dev), y_dev) # accuracy from validation set to avoid overfitting
        accuracies.append([acc, state, gmm])

# %%
accuracies

# %%
for i in range(len(accuracies)):
    print("{:.4f}".format(accuracies[i][0]))

# %% [markdown]
# In each element of the list "accuracies" the first is the accuracy the second the number of states and the third one the number of gaussian distirbutions.

# %%
acc = []
hmm_states = []
gmm_dist = []
max_acc = 0

for accu in accuracies:
    acc.append(accu[0])
    hmm_states.append(accu[1])
    gmm_dist.append(accu[2])
    if accu[0] > max_acc:
        max_acc = accu[0]
        max_hmm = accu[1]
        max_gmm = accu[2]

# %%
print("The best accuracy we have achieved is {:.3f} while using models with {} HMM states and {} Gaussian Distirbutions.".format(max_acc, max_hmm, max_gmm))

# %%
k = 0
shape = (len(states), len(GMMs))
plot_acc = np.empty(shape)

for i in range(len(states)):
    for z in range(len(GMMs)):
        plot_acc[i][z] = acc[k]
        k = k + 1

# %%
X = states
Y = GMMs
Z = plot_acc

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(Y, X, Z, 100)
ax.set_xlabel('Number of GMM distirbutions')
ax.set_ylabel('Number of hmm states')
ax.set_zlabel('accuracy');

# %% [markdown]
# ## Step 13 - Confusion Matrix

# %%
print('max_hmm = {}'.format(max_hmm))
print('max_gmm = {}'.format(max_gmm))

# %% [markdown]
# Making the models with {{ max_hmm }} HMM states and {{ max_gmm }} Gaussian Distirbutions.

# %%
models = []
for i in range(10):
    models.append(train_GMM_HMM(X_train_per_digit[i], n_states = max_hmm, n_mixtures = max_gmm))

# %% [markdown]
# plot_confusion_matrix is a function that allows us to plot the Confusion Matrix.

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# %% [markdown]
# Saving the predictions for our models and calculating the elements of the confusion matrix.

# %% [markdown]
# Confusion Matrix for validation set.

# %%
preds = predictions(models, X_dev)

Confusion_Matrix = np.zeros((10,10))

for sample in range(len(preds)):
    Confusion_Matrix[y_dev[sample], preds[sample]] += 1
    
Confusion_Matrix = Confusion_Matrix.astype(int)

# %%
print("The accuracy for the validation set is {:.3f}.".format(accuracy(predictions(models, X_dev), y_dev)))
plot_confusion_matrix(Confusion_Matrix, classes = np.unique(y_dev), normalize=False, title='Confusion matrix\nValidation Set', cmap=plt.cm.Blues)

# %% [markdown]
# Confusion Matrix for test set.

# %%
preds = predictions(models, X_test)

Confusion_Matrix = np.zeros((10,10))

for sample in range(len(preds)):
    Confusion_Matrix[y_test[sample], preds[sample]] += 1
    
Confusion_Matrix = Confusion_Matrix.astype(int)

# %%
print("The accuracy for the test set is {:.3f}.".format(accuracy(predictions(models, X_test), y_test)))
plot_confusion_matrix(Confusion_Matrix, classes = np.unique(y_test), normalize=False, title='Confusion matrix\nTest Set', cmap=plt.cm.Reds)

# %% [markdown]
# ### Βήμα 14 - LSTM Models

# %%
#code from lstm.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths = [len(instance) for instance in feats] # Find the lengths 

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #
        padded=([np.pad(instance, ((0, np.max(self.lengths)-len(instance)), (0, 0))) for instance in x])
        return np.array(padded)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)

# %%
# define parameters, dataset and dataloader for all 3 sets
batch_size = 64
epochs = 100
crit = nn.CrossEntropyLoss()

train_set = FrameLevelDataset(X_train, y_train)
val_set = FrameLevelDataset(X_dev, y_dev)
test_set = FrameLevelDataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# %%
class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False,dropout=0,bonus=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.bonus=bonus
        self.rnn_size = rnn_size
        self.dropout = dropout
        if(self.bidirectional==True):
            self.num_layers=2*num_layers
        else:
            self.num_layers=num_layers
        self.lstm = nn.LSTM(input_dim, rnn_size, num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        # initialize
        h0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size)

        if (self.bonus==True):

        # sort sequences by decreasing length
            lengths, indx = lengths.sort(dim=0, descending=True)
            x_pck = pack_padded_sequence(x[indx],
                                            list(lengths.data),
                                            batch_first=True,
                                            enforce_sorted = True)
            lstm_output_1, _ = self.lstm(x_pck, (h0, c0))
            lstm_output_2 = pad_packed_sequence(lstm_output_1, batch_first=True)[0]
            last_outputs = self.fc(self.last_timestep(lstm_output_2, lengths, self.bidirectional))
            return last_outputs,indx
        else:
            lstm_output, _ = self.lstm(x, (h0, c0))
            last_outputs = self.fc(self.last_timestep(lstm_output, lengths, self.bidirectional))
            return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()



# define a new training function
def train(model, dataloader, criterion, optimizer):    
    tr_loss = 0.0
    model.train()   # train mode
    for i, batch in enumerate(dataloader, 1):
        inputs, labels, lengths = batch
        optimizer.zero_grad()   # zero gradients out
        y_preds = model(inputs, lengths)    # forward propagate
        loss = criterion(y_preds, labels)   # compute loss function
        loss.backward() # compute gradients
        optimizer.step()    # update parameters

        tr_loss += loss.data.item()

    return tr_loss/i   # train loss


# neural network evaluator
def eval(model, dataloader, criterion):
    val_loss = 0.0
    y_act = []
    y_pred = []
    
    model.eval() # evaluation mode
    with torch.no_grad(): # do not compute gradients
        for i, batch in enumerate(dataloader, 1):
            
            inputs, labels, lengths = batch
                
            y_preds = model(inputs, lengths) #forward propagate
            loss = criterion(y_preds, labels) #loss function
            pred = torch.argmax(y_preds, dim=1) #make predictions
            
            y_pred.append(pred.numpy()) #get predicted labels
            y_act.append(labels.numpy()) #get real lables

            val_loss += loss.data.item() #add to validation loss

    return val_loss / i, (y_act, y_pred)

     


# %%
#training and validation loss
LSTM1 = BasicLSTM(input_dim=X_train[0].shape[1], rnn_size=128, output_dim=10, num_layers=1, bidirectional=False, bonus=False,dropout=0)
opt = torch.optim.Adam(LSTM1.parameters(), lr=0.001)
train_arr_1=[]
val_arr_1=[]
y_pred_1=[]
y_val_1=[]
for i in range(epochs):
    temp_tr=train(LSTM1,train_loader,criterion=crit,optimizer=opt)
    temp_val=eval(LSTM1,val_loader,criterion=crit)
    val_arr_1.append(temp_val[0])
    train_arr_1.append(temp_tr)
    print("Epoch {}: Training Loss = {} Validation Loss = {}".format(i, temp_tr,temp_val[0]))
    torch.save(LSTM1, "./LSTM1") # checkpoint




# %%
#print losses and accuracies of LSTM1
_,(val_real_1,val_1)=eval(LSTM1,val_loader,criterion=crit)
_,(test_real_1,test_1)=eval(LSTM1,test_loader,criterion=crit)
acc_val_1=accuracy_score(np.concatenate(val_1),np.concatenate(val_real_1))
acc_test_1=accuracy_score(np.concatenate(test_1),np.concatenate(test_real_1))
print("Simple LSTM Accuracies: Validation Set: {} % -- Test Set: {} %".format(acc_val_1*100,acc_test_1*100))
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
plt.plot(np.arange(epochs),train_arr_1,label='Train Loss',color='b')
plt.plot(np.arange(epochs),val_arr_1,label='Validation Loss',color='y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Simple LSTM Losses')
plt.show()

# %%
#training and validation loss with dropout and wd
LSTM2 = BasicLSTM(input_dim=X_train[0].shape[1], rnn_size=128, output_dim=10, num_layers=1, bidirectional=False, bonus=False,dropout=0.2)
opt = torch.optim.Adam(LSTM2.parameters(), lr=0.001,weight_decay=1e-4) #l2 regulization
train_arr_2=[]
val_arr_2=[]
for i in range(epochs):
    temp_tr=train(LSTM2,train_loader,criterion=crit,optimizer=opt)
    temp_val=eval(LSTM2,val_loader,criterion=crit)
    val_arr_2.append(temp_val[0])
    train_arr_2.append(temp_tr)
    print("Epoch {}: Training Loss = {} Validation Loss = {}".format(i, temp_tr,temp_val[0]))
    torch.save(LSTM2, "./LSTM2") # checkpoint

# %%
#print losses and accuracies of LSTM2
_,(val_real_2,val_2)=eval(LSTM2,val_loader,criterion=crit)
_,(test_real_2,test_2)=eval(LSTM2,test_loader,criterion=crit)
acc_val_2=accuracy_score(np.concatenate(val_2),np.concatenate(val_real_2))
acc_test_2=accuracy_score(np.concatenate(test_2),np.concatenate(test_real_2))
print("Simple LSTM (with dropout and L2 reg.) Accuracies: Validation Set: {} % -- Test Set: {} %".format(acc_val_2*100,acc_test_2*100))
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
plt.plot(np.arange(epochs),train_arr_2,label='Train Loss',color='b')
plt.plot(np.arange(epochs),val_arr_2,label='Validation Loss',color='y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Simple LSTM(with dropout and L2 reg.) Losses')

# %%
#plot confusion matrix for LSTM with best accuracy scores
best_lstm=torch.load("./LSTM2")
_,(val_real_best,val_best)=eval(best_lstm,val_loader,criterion=crit)
_,(test_real_best,test_best)=eval(best_lstm,test_loader,criterion=crit)

best_cm_val = np.zeros((10,10))
best_cm_test = np.zeros((10,10))

for sample in range(len(np.concatenate(val_best))):
    best_cm_val[np.concatenate(val_real_best)[sample], np.concatenate(val_best)[sample]] += 1

for sample in range(len(np.concatenate(test_best))):
    best_cm_test[np.concatenate(test_real_best)[sample], np.concatenate(test_best)[sample]] += 1

plot_confusion_matrix(best_cm_val, classes = [i for i in range(10)], normalize=True, title='Confusion matrix of best LSTM (Val.Set)', cmap=plt.cm.Blues)



# %%
plot_confusion_matrix(best_cm_test, classes = [i for i in range(10)], normalize=True, title='Confusion matrix of best LSTM (Test Set)', cmap=plt.cm.Reds)

# %%
#training and validation loss with dropout and wd and early stopping
LSTM3 = BasicLSTM(input_dim=X_train[0].shape[1], rnn_size=128, output_dim=10, num_layers=1, bidirectional=False, bonus=False,dropout=0.2)
opt = torch.optim.Adam(LSTM3.parameters(), lr=0.001,weight_decay=1e-4)
train_arr_3=[]
val_arr_3=[]
counter=0.0
min_loss=np.inf
for i in range(epochs):
    temp_tr=train(LSTM3,train_loader,criterion=crit,optimizer=opt)
    temp_val=eval(LSTM3,val_loader,criterion=crit)
    val_arr_3.append(temp_val[0])
    train_arr_3.append(temp_tr)
    print("Epoch {}: Training Loss = {} Validation Loss = {}".format(i, temp_tr,temp_val[0]))
    if (temp_val[0] < min_loss) :
        torch.save(LSTM3, "./LSTM3") # checkpoint
        min_loss = temp_val[0]
        counter = 0
    else:
        counter += 1
    
    if counter == 10:  # if it doesn't get any better, stop so as not to overfit
        print("Early Stopping")
        break

# %%
#print losses and accuracies of LSTM3
_,(val_real_3,val_3)=eval(LSTM3,val_loader,criterion=crit)
_,(test_real_3,test_3)=eval(LSTM3,test_loader,criterion=crit)
acc_val_3=accuracy_score(np.concatenate(val_3),np.concatenate(val_real_3))
acc_test_3=accuracy_score(np.concatenate(test_3),np.concatenate(test_real_3))
print("LSTM with Early Stopping Accuracies: Validation Set: {} % -- Test Set: {} %".format(acc_val_3*100,acc_test_3*100))
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
plt.plot(np.arange(np.shape(train_arr_3)[0]),train_arr_3,label='Train Loss',color='b')
plt.plot(np.arange(np.shape(val_arr_3)[0]),val_arr_3,label='Validation Loss',color='y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('LSTM (with Early Stopping) Losses')
plt.show()

# %%
#training and validation loss with dropout and wd and early stopping and bd
LSTM4 = BasicLSTM(input_dim=X_train[0].shape[1], rnn_size=128, output_dim=10, num_layers=1, bidirectional=True, bonus=False,dropout=0.2)
opt = torch.optim.Adam(LSTM4.parameters(), lr=0.001,weight_decay=1e-4)
train_arr_4=[]
val_arr_4=[]
counter=0.0
min_loss=np.inf
for i in range(epochs):
    temp_tr=train(LSTM4,train_loader,criterion=crit,optimizer=opt)
    temp_val=eval(LSTM4,val_loader,criterion=crit)
    val_arr_4.append(temp_val[0])
    train_arr_4.append(temp_tr)
    print("Epoch {}: Training Loss = {} Validation Loss = {}".format(i, temp_tr,temp_val[0]))
    if (temp_val[0] < min_loss) :
        torch.save(LSTM4, "./LSTM4") # checkpoint
        min_loss = temp_val[0]
        counter = 0
    else:
        counter += 1
    
    if counter == 10:  # if it doesn't get any better, stop so as not to overfit
        print("Early Stopping")
        break

# %%
#print losses and accuracies of LSTM4
_,(val_real_4,val_4)=eval(LSTM4,val_loader,criterion=crit)
_,(test_real_4,test_4)=eval(LSTM4,test_loader,criterion=crit)
acc_val_4=accuracy_score(np.concatenate(val_4),np.concatenate(val_real_4))
acc_test_4=accuracy_score(np.concatenate(test_4),np.concatenate(test_real_4))
print("Bidirectional LSTM: Validation Set: {} % -- Test Set: {} %".format(acc_val_4*100,acc_test_4*100))
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
plt.plot(np.arange(np.shape(train_arr_4)[0]),train_arr_4,label='Train Loss',color='b')
plt.plot(np.arange(np.shape(val_arr_4)[0]),val_arr_4,label='Validation Loss',color='y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Bidirectional LSTM Losses')
plt.show()

# %%
#training and validation loss with dropout and wd and early stopping and bd and bonus
# define a new training function
def train5(model, dataloader, criterion, optimizer):    
    tr_loss = 0.0
    model.train()   # train mode
    for i, batch in enumerate(dataloader, 1):
        inputs, labels, lengths = batch
        optimizer.zero_grad()   # zero gradients out
        y_preds,indices = model(inputs, lengths)    # forward propagate
        loss = criterion(y_preds, labels[indices])   # compute loss function
        loss.backward() # compute gradients
        optimizer.step()    # update parameters

        tr_loss += loss.data.item()

    return tr_loss/i   # train loss


# neural network evaluator
def eval5(model, dataloader, criterion):
    val_loss = 0.0
    y_act = []
    y_pred = []
    
    model.eval() # evaluation mode
    with torch.no_grad(): # do not compute gradients
        for i, batch in enumerate(dataloader, 1):
            
            inputs, labels, lengths = batch
                
            y_preds,indices = model(inputs, lengths) #forward propagate
            loss = criterion(y_preds, labels[indices]) #loss function
            pred = torch.argmax(y_preds, dim=1) #make predictions
            
            y_pred.append(pred.numpy()) #get predicted labels
            y_act.append(labels.numpy()) #get real lables

            val_loss += loss.data.item() #add to validation loss

    return val_loss / i, (y_act, y_pred)

# neural network evaluator
def eval(model, dataloader, criterion):
    val_loss = 0.0
    y_act = []
    y_pred = []
    
    model.eval() # evaluation mode
    with torch.no_grad(): # do not compute gradients
        for i, batch in enumerate(dataloader, 1):
            
            inputs, labels, lengths = batch
                
            y_preds = model(inputs, lengths) #forward propagate
            loss = criterion(y_preds, labels) #loss function
            pred = torch.argmax(y_preds, dim=1) #make predictions
            
            y_pred.append(pred.numpy()) #get predicted labels
            y_act.append(labels.numpy()) #get real lables

            val_loss += loss.data.item() #add to validation loss

    return val_loss / i, (y_act, y_pred)



LSTM5 = BasicLSTM(input_dim=X_train[0].shape[1], rnn_size=128, output_dim=10, num_layers=1, bidirectional=True, bonus=True,dropout=0.2)
opt = torch.optim.Adam(LSTM5.parameters(), lr=0.001,weight_decay=1e-4)
train_arr_5=[]
val_arr_5=[]
counter=0.0
min_loss=np.inf
for i in range(epochs):
    temp_tr=train5(LSTM5,train_loader,criterion=crit,optimizer=opt)
    temp_val=eval5(LSTM5,val_loader,criterion=crit)
    val_arr_5.append(temp_val[0])
    train_arr_5.append(temp_tr)
    print("Epoch {}: Training Loss = {} Validation Loss = {}".format(i, temp_tr,temp_val[0]))
    if (temp_val[0] < min_loss) :
        torch.save(LSTM5, "./LSTM5") # checkpoint
        min_loss = temp_val[0]
        counter = 0
    else:
        counter += 1
    
    if counter == 10:  # if it doesn't get any better, stop so as not to overfit
        print("Early Stopping")
        break

# %%
#print losses and accuracies of LSTM5
_,(val_real_5,val_5)=eval5(LSTM5,val_loader,criterion=crit)
_,(test_real_5,test_5)=eval5(LSTM5,test_loader,criterion=crit)
acc_val_5=accuracy_score(np.concatenate(val_5),np.concatenate(val_real_5))
acc_test_5=accuracy_score(np.concatenate(test_5),np.concatenate(test_real_5))
print("LSTM with pack_padded_sequence: Validation Set: {} % -- Test Set: {} %".format(acc_val_5*100,acc_test_5*100))
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)
plt.plot(np.arange(np.shape(train_arr_5)[0]),train_arr_5,label='Train Loss',color='b')
plt.plot(np.arange(np.shape(val_arr_5)[0]),val_arr_5,label='Validation Loss',color='y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM with pack_padded_sequence Losses')
plt.legend()
plt.show()


