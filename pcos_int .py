#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("PCOS_data.csv")


# In[2]:


df.head()


# In[3]:


df.info()


# In[ ]:





# In[4]:


df = df.rename(columns=lambda x: x.strip())


# In[5]:


df.columns


# In[ ]:





# In[6]:


df.head()


# In[7]:


df.isna().sum()


# In[8]:


marriage_status_mean = df['Marraige Status (Yrs)'].mean()
fast_food_mean = df['Fast food (Y/N)'].mean()
df['Marraige Status (Yrs)'].fillna(marriage_status_mean, inplace=True)
df['Fast food (Y/N)'].fillna(fast_food_mean, inplace=True)


# In[9]:


X= df.drop("PCOS (Y/N)", axis=1)
y= df["PCOS (Y/N)"]


# In[10]:


plt.figure(figsize=(14,8))
sns.heatmap(df.isna().transpose(),
            cmap="winter",
            cbar_kws={'label': 'Missing Data'}, xticklabels=True, yticklabels=True)


# In[ ]:





# In[11]:


df["PCOS (Y/N)"].value_counts()


# In[12]:


#Classsification using Linear Regression


# In[13]:


df


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

# List of models to compare
models = [
    LogisticRegression(max_iter=10000), 
    SVC(), 
    KNeighborsClassifier(), 
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(eval_metric='logloss'),
    CatBoostClassifier(verbose=0)  
]

def compare_models_cross_validation(X, y):
    for model in models:
        cv_score = cross_val_score(model, X, y, cv=5)
        mean_accuracy = cv_score.mean() * 100
        mean_accuracy = round(mean_accuracy, 2)

        print(f'Cross Validation accuracies for {model.__class__.__name__} = {cv_score}')
        print(f'Accuracy score of the {model.__class__.__name__} = {mean_accuracy}%')
        print('---------------------------------------------------------------')

# Assuming X and y are already defined
# Example usage:
# compare_models_cross_validation(X, y)


# In[15]:


compare_models_cross_validation(X,y)


# In[16]:


pip install catboost


# In[17]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=10,
    depth=3,
    learning_rate=0.1,
    loss_function='MultiClass'
)

# Train the model
model.fit(X_train, y_train, verbose=False)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Convert the predictions to integers
y_pred = y_pred.astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of CatBoost classifier on test set:', accuracy)

# Compute and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nThe confusion Matrix is : \n', conf_matrix)

# Print classification report
print('\nThe evaluation parameters are : \n', classification_report(y_test, y_pred))


# In[18]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

def ModelSelection(list_of_models, hyperparameters_dictionary, dataset):
    result = []

    # Load dataset and separate features and target
    X = dataset.data
    Y = dataset.target

    # Extract model names
    model_keys = list(hyperparameters_dictionary.keys())

    for model in list_of_models:
        # Get the class name of the model
        model_name = model.__class__.__name__

        # Ensure the model name is in the hyperparameters dictionary
        if model_name not in model_keys:
            print(f"No hyperparameters found for {model_name}. Skipping this model.")
            continue

        # Update the hyperparameters keys to fit the pipeline format
        params = {f'model__{k}': v for k, v in hyperparameters_dictionary[model_name].items()}

        print(f"Model: {model_name}")
        print(f"Hyperparameters: {params}")
        print('---------------------------------')

        # Create a pipeline with StandardScaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        classifier = GridSearchCV(pipeline, params, cv=5)

        # Fitting the data to classifier
        classifier.fit(X, Y)

        result.append({
            'model used': model_name,
            'highest score': classifier.best_score_,
            'best hyperparameters': classifier.best_params_
        })

    result_dataframe = pd.DataFrame(result, columns=['model used', 'highest score', 'best hyperparameters'])

    return result_dataframe

# Define your models
models_list = [
    LogisticRegression(max_iter=10000), 
    SVC(), 
    KNeighborsClassifier(), 
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(eval_metric='logloss'),
    CatBoostClassifier(verbose=0)  # Adding CatBoostClassifier
]

# Define hyperparameters for each model
model_hyperparameters = {
    'LogisticRegression': {'C': [1, 5, 10, 20]},
    'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1, 5, 10, 20]},
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 10]},
    'RandomForestClassifier': {'n_estimators': [10, 20, 50, 100]},
    'DecisionTreeClassifier': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'XGBClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'CatBoostClassifier': {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [4, 6, 8]}  # Adding hyperparameters for CatBoost
}

# Load sample data
data = load_iris()

# Call the function
result = ModelSelection(models_list, model_hyperparameters, data)
print(result)


# In[19]:


df.shape


# In[20]:


scaler = StandardScaler()


# In[21]:


scaler.fit(X)


# In[22]:


standardized_data = scaler.transform(X)


# In[23]:


print(standardized_data)


# In[24]:


X= df.drop("PCOS (Y/N)", axis=1)
y= df["PCOS (Y/N)"]


# In[25]:


print(X)
print(y)


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, stratify=y, random_state=2)


# In[27]:


print(X.shape, X_train.shape, X_test.shape)


# In[28]:


from sklearn import svm


# In[29]:


classifier = svm.SVC(kernel='linear')


# In[30]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[31]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[32]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[33]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[34]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[ ]:





# In[35]:


input_data = (2,2,36,65,161.5,24.9,15,74,20,11.7,2,5,11,1,0,38,32,0.84,0,0,0,0,0,0,0,120)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have pcos')
else:
    print('The person have pcos')


# In[36]:


pip install gradio


# In[41]:


import pandas as pd
import numpy as np
import tensorflow as tf
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load and preprocess data
df = pd.read_csv("PCOS_data.csv")
df = df.rename(columns=lambda x: x.strip())
df['Marraige Status (Yrs)'].fillna(df['Marraige Status (Yrs)'].mean(), inplace=True)
df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].mean(), inplace=True)
X = df.drop("PCOS (Y/N)", axis=1)
y = df["PCOS (Y/N)"]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Gradio interface function for numerical data
def predict_pcos(*input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(input_data_scaled)
    if prediction[0] == 0:
        return 'The person does not have PCOS'
    else:
        return 'The person has PCOS'

# Define the input fields based on your dataset features
input_fields = [
    gr.Number(label="Sl.No"),
    gr.Number(label="Patient File No."),
    gr.Number(label="Age (yrs)"),
    gr.Number(label="Weight (Kg)"),
    gr.Number(label="Height (Cm)"),
    gr.Number(label="BMI"),
    gr.Number(label="Blood Group"),
    gr.Number(label="Pulse rate (bpm)"),
    gr.Number(label="RR (breaths/min)"),
    gr.Number(label="Hb (g/dl)"),
    gr.Number(label="Cycle (R/I)"),
    gr.Number(label="Cycle length (days)"),
    gr.Number(label="Marraige Status (Yrs)"),
    gr.Number(label="Pregnant (Y/N)"),
    gr.Number(label="No. of abortions"),
    gr.Number(label="Hip (inch)"),
    gr.Number(label="Waist (inch)"),
    gr.Number(label="Waist:Hip Ratio"),
    gr.Number(label="Weight gain (Y/N)"),
    gr.Number(label="Hair growth (Y/N)"),
    gr.Number(label="Skin darkening (Y/N)"),
    gr.Number(label="Hair loss (Y/N)"),
    gr.Number(label="Pimples (Y/N)"),
    gr.Number(label="Fast food (Y/N)"),
    gr.Number(label="Reg. Exercise (Y/N)"),
    gr.Number(label="BP Systolic (mmHg)")
]

# Load the pre-trained image classification model
model = tf.keras.models.load_model('model1.h5')

# Define the class names
class_names = ['pcos', 'normal']

# Gradio interface function for image data
def predict_image(image):
    # Resize the image to the expected input shape of the model
    image_resized = tf.image.resize(image, (224, 224))

    # Normalize the image to [0, 1] range
    image_resized = image_resized / 255.0

    # Expand dimensions to add batch size dimension
    image_resized = tf.expand_dims(image_resized, axis=0)

    # Make a prediction
    pred = model.predict(image_resized)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(pred)

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# Create a Gradio interface with tabs for both functionalities
with gr.Blocks() as demo:
    gr.Markdown("# PCOS Detection")
    gr.Markdown("This application can predict PCOS based on numerical data or image data.")
    
    with gr.Tab("Numerical Data"):
        gr.Interface(
            fn=predict_pcos,
            inputs=input_fields,
            outputs=gr.Textbox(label="Prediction"),
            live=True
        ).render()
    
    with gr.Tab("Image Data"):
        gr.Interface(
            fn=predict_image,
            inputs=gr.Image(type="numpy", label="Upload an Image"),
            outputs=gr.Textbox(label="Prediction"),
            live=True
        ).render()

# Launch the combined interface
demo.launch()


# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load and preprocess data
df = pd.read_csv("PCOS_data.csv")
df = df.rename(columns=lambda x: x.strip())
df['Marraige Status (Yrs)'].fillna(df['Marraige Status (Yrs)'].mean(), inplace=True)
df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].mean(), inplace=True)
X = df.drop("PCOS (Y/N)", axis=1)
y = df["PCOS (Y/N)"]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Gradio interface function for numerical data
def predict_pcos(*input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(input_data_scaled)
    if prediction[0] == 0:
        return 'The person does not have PCOS'
    else:
        return 'The person has PCOS'

# Define the input fields based on your dataset features
input_fields = [
    gr.Number(label="Sl.No"),
    gr.Number(label="Patient File No."),
    gr.Number(label="Age (yrs)"),
    gr.Number(label="Weight (Kg)"),
    gr.Number(label="Height (Cm)"),
    gr.Number(label="BMI"),
    gr.Number(label="Blood Group"),
    gr.Number(label="Pulse rate (bpm)"),
    gr.Number(label="RR (breaths/min)"),
    gr.Number(label="Hb (g/dl)"),
    gr.Number(label="Cycle (R/I)"),
    gr.Number(label="Cycle length (days)"),
    gr.Number(label="Marraige Status (Yrs)"),
    gr.Number(label="Pregnant (Y/N)"),
    gr.Number(label="No. of abortions"),
    gr.Number(label="Hip (inch)"),
    gr.Number(label="Waist (inch)"),
    gr.Number(label="Waist:Hip Ratio"),
    gr.Number(label="Weight gain (Y/N)"),
    gr.Number(label="Hair growth (Y/N)"),
    gr.Number(label="Skin darkening (Y/N)"),
    gr.Number(label="Hair loss (Y/N)"),
    gr.Number(label="Pimples (Y/N)"),
    gr.Number(label="Fast food (Y/N)"),
    gr.Number(label="Reg. Exercise (Y/N)"),
    gr.Number(label="BP Systolic (mmHg)")
]

# Load the pre-trained image classification model
model = tf.keras.models.load_model('model1.h5')

# Define the class names
class_names = ['pcos', 'normal']

# Gradio interface function for image data
def predict_image(image):
    # Resize the image to the expected input shape of the model
    image_resized = tf.image.resize(image, (224, 224))

    # Normalize the image to [0, 1] range
    image_resized = image_resized / 255.0

    # Expand dimensions to add batch size dimension
    image_resized = tf.expand_dims(image_resized, axis=0)

    # Make a prediction
    pred = model.predict(image_resized)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(pred)

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# Create a Gradio interface with tabs for both functionalities
with gr.Blocks() as demo:
    gr.Markdown("# INSIGHTS OF PCOS")
    gr.Markdown("This application can predict and detect PCOS based on numerical data or image data.")
    
    with gr.Tab("PREDICTION"):
        gr.Interface(
            fn=predict_pcos,
            inputs=input_fields,
            outputs=gr.Textbox(label="Prediction"),
            live=False
        )
    
    with gr.Tab("DETECTION"):
        gr.Interface(
            fn=predict_image,
            inputs=gr.Image(type="numpy", label="Upload an Image"),
            outputs=gr.Textbox(label="Prediction"),
            live=False
        )
        

# Launch the combined interface
demo.launch()


# In[ ]:





# In[ ]:




