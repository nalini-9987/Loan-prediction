import pickle
import streamlit as st
# from PIL import Image

# loading the trained model
pickle_in = open('har.pkl', 'rb')
classifier = pickle.load(pickle_in)


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,Dependents,Property_Area):
    # Pre-processing user input
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1

    if Credit_History == "Clear Debts":
        Credit_History = 1
    else:
        Credit_History = 0

    if Property_Area == "Rural":
        Property_Area = 0
    elif Property_Area == "Semiurban":
        Property_Area = 1
    else:
        Property_Area = 2

    LoanAmount = LoanAmount / 1000

    # Making predictions
    prediction = classifier.predict(
        [[Gender, Married, ApplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History,Dependents,Property_Area]])

    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Loan Prediction </h1> 
    </div> 
    """
    # image = Image.open('lppppp.png')
    # st.image(image, caption='Loan Prediction')

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    ApplicantIncome = st.slider("Applicants monthly income",150,81000)
    Loan_Amount_Term = st.slider("Applicants loan amount term",12,480)
    LoanAmount = st.slider("Total loan amount",9,700)
    Credit_History = st.selectbox('Credit_History', ("Clear Debts", "Unclear Debts"))
    Dependents = st.selectbox("no of dependents",("0","1","2","3","4","heroku login"))
    Property_Area = st.radio("Property Land",("Rural","Semiurban","Urban"))

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount,Loan_Amount_Term,Credit_History,Dependents, Property_Area)
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)


if __name__ == '__main__':
    main()