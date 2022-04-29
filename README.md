# Automating-Loan-Prediction
Building a web app using streamlit and deploying it to the cloud with Heroku.

# Table of Contents
1. Introduction 
2. Problem Statememt
3. Data Cleaning and Processing
4. Data Visualization
5. Building a Machine Learning Model
6. Building the app using python library 'Streamlit'
7. Deployment via Heroku
8. Conclusion

# Introduction
In the tech world, the ultimate goal of any project is to deploy it into production. And with numerous advanced technologies, it has become extremely easier to 
execute and experiment a lot of such stuffs. Likewise, in the data science domain, while building any machine learning model whether to predict sales or any other 
outcome or to identify any objects or whatever be the reason, the main objective of the project is to be made available to the end users. However, different 
organizations deal with deployment in different ways. Some companies might have completely different teams for the deployment part, while sometimes it is the sole 
responsibility of the data scientists themselves to involve in end to end machine learning projects. So, I am writing this article to understand how to build a 
simple web app for a simple machine learning problem and deploy it to the cloud for our end users.

# Problem Statement
A Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are 
Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to 
identify the customers’ segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data 
set. Now we have finalized the problem statement, generated the hypotheses, and collected the data. Next are the Data exploration and pre-processing phase.

# Building a Machine Learning Model
In this article, I will not go much into detail about building the machine learning model. However, I will just list down the steps needed to create a model. 
Import all the required libraries. Read the dataset. Data Cleansing. Understand the data - Exploratory data analysis - Visual data analysis - Label Encoding - Split 
the data into train and test. Build the models on the train data try out multiple models and chose the one with the best accuracy. Use the trained model to predict 
on the test data.

# Saving the Machine Learning Model
Let us understand the need of saving a machine learning model. Suppose, in our example problem statement, I have used JuPyter Notebook for building my model to 
help me with the predictions on any new data. So, if I want to do the predictions on another day or some other time, I have to run the whole notebook again. Or, 
if my friends or colleagues want to do the same prediction, then I will have to pass my entire notebook to them for their use. Hence, instead of running the entire
code again , we can just save and load the model whenever and wherever required. It is also called dumping the model as we are storing information such as 
coefficients and other related data about the model. It basically indicates re-usability of our code. There are numerous python packages to save a machine learning
model like . Pickle . Joblib However , I have used Pickle module for my task. So, once we finalize the model with the best accuracy , we can save the model on the 
disk with the pickle module in python by first importing it. The back-end code for the data processing and the model building is available in github.

# Building the App using python library 'Streamlit'
Whenever we think about building a web app, all that comes to our mind are HTML, JavaScript and almost all the companies have designated teams completely comprised
of front end web developers. Since data scientists are more concerned about the processing of the data , the back-end development of models which itself is 
extremely time consuming, it might not be easier for a data scientist to again invest time and energy on front end development. However, there is this python 
library called ‘Streamlit’ which comes as relief to a data scientist as Streamlit is one of the easiest ways to build a front end web app for our machine learning 
projects with simple python scripts. And all these without any HTML or Javascript. Streamlit is an open source python library similar to a wrapper class. We all 
know what a wrapper class is. A simple definition of wrapper class is any class which “wraps” or “encapsulates” the functionality of another class or component. 
Streamlit behaves in the exact same way. We load the model and pass the features entered by the end user back to our model based on which all the processing is 
done on the back end and finally we send the result/prediction back to the web app.

# Deployment via Heroku
Heroku is a container-based cloud Platform as a Service (PaaS) where we can easily deploy web apps for our end users to use. What is the need for deployment: A 
simple explanation: Till now , my app runs only on my local system. So, if someone else wants to use my app, I would have to forward him/her my notebooks with all 
the code, then they will need to run the entire code again, save it in their system, and again run it in their local system to use the app .Lengthy process right?

# Conclusion
In this project we have examined on Automating Loan Prediction Data by performing step by step operation of EDA/VDA. Here on, applying machine learning Logistic 
Regression  Model and Random Forest Classifier Model. Considering the Best Accuracy Score building a streamlit page then heroku deploying on cloud platform (Heroku).

# Heroku Webapp Link
https://hloanprediction.herokuapp.com/
