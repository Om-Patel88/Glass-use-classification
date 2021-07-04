import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feat_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
def prediction(model,feat_cols):
  pred = model.predict(feat_cols)
  if pred == 1:
    return "building windows float processed".upper()
  elif pred == 2:
    return "building windows non float processed".upper()
  elif pred == 3:
    return "vehicle windows float processed".upper()
  elif pred == 4:
    return "vehicle windows non float processed".upper()
  elif pred ==5:
    return "containers".upper()
  elif pred == 6:
    return "tableware".upper()

st.title("Glass type Prediction Web-App")
st.sidebar.title("Glass type Prediction Web-App")

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Glass-Type Datasets")
  st.dataframe(glass_df)
  
# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader("Visualisation Selector")
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot').
# Store the current value of this widget in a variable 'plot_list'.
plot_list = st.sidebar.multiselect("Select the Charts/Plots:",('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))
# Display line chart 
if 'Line Chart' in plot_list:
  st.subheader("Line Chart")
  st.line_chart(glass_df)



# Display area chart    
if 'Area Chart' in plot_list:
  st.subheader("Area Chart")
  st.area_chart(glass_df)
  
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Display correlation heatmap using seaborn module and 'st.pyplot()'
if 'Correlation Heatmap' in plot_list:
  st.subheader("Correlation Heat Map")
  plt.figure(figsize=(12,12))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()
# Display count plot using seaborn module and 'st.pyplot()' 
if 'Count Plot' in plot_list:
  st.subheader("Count Plot")
  plt.figure(figsize=(5,10))
  sns.countplot(x='GlassType',data=glass_df)
  st.pyplot()
# Display pie chart using matplotlib module and 'st.pyplot()'   
if 'Pie Chart' in plot_list:
  st.subheader("Pie Plot")
  pie_data = glass_df['GlassType'].value_counts()
  plt.figure(figsize=(5,5))
  plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .12, 6))
  st.pyplot()
if 'Box Plot' in plot_list:
  st.subheader('Box Plot')
  column = st.sidebar.selectbox("Select the column for boxplot", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize=(12,2))
  sns.boxplot(glass_df[column])
  st.pyplot()
