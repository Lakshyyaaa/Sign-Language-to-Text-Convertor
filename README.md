# Sign-Language-to-Text-Convertor
Converts Sign Language to Text as Initialised by the user. The project was created for SIH 2024 Problem Statement Number 1716.

The Project is divided into four parts -->
1. Capturing the Sign Languages
2. Converting the Captured Sign Languages into DataSets
3. Training the Captured Data Sets
4. Testing the Data Set

**Capturing the Sign language (file_name: image_collection.py)**
The first code captures n number of Sign Language gestures with each gesture having 100 frames clubbed in a directory 
 
**Converting the Captured Sign Language to DataSets (file_name:create_dataset.py)**
The second code converts the Sign Language Directory captured in the first code to a DataSet using the Pickle Module of Python

**Training the Captured Data Sets (file_name: data_training.py)**
The third code refers to the data set created in second code refering the hand points using mediapipe library in Python and training the model to response accordingly

**Testing the Data Set** 
The fourth code the final code tests the trained data set and converts the hand gestures as intialised in the first code to text language and copies the same to a txt file

Libraries and Modules Needed --> 
  - open cv (cv2)
  - pickle
  - scikit learn (RandomForestClassifier, accuracy_score, train_test_split)
  - numpy
  - mediapipes
  - warning (if SymbolDatabase.GetPrototype() is used)
  - time 
