import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import base64
from streamlit_autorefresh import st_autorefresh
from PIL import Image


st.set_page_config(page_title="Fraud Detection System", layout="wide")

model = pickle.load(open("model.pkl", "rb"))


if 'reported_transactions' not in st.session_state:
    st.session_state.reported_transactions = []

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'expander_open' not in st.session_state:
    st.session_state.expander_open = False


st.markdown("""
    <div style='text-align: center;'>
        <h1 style='white-space: nowrap;'>Online Payments Fraud Detection System</h1>
    </div>
    """, unsafe_allow_html=True)


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.ibb.co/68MXp2t/background-img.jpgg");
background-size: 220%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


def streamlit_menu():
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Single-Predict", "File-Predict" ,"History", "About"],  # required
        icons=["h", "u","p", "l", "i",],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "nav-link-selected": {"background-color": "blue"},
        }
    )
    return selected

selected = streamlit_menu()



if selected == "Home":

    st.markdown("<h2 style='text-align: center; color: yellow'>Detect Fraudulent Transactions</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        images = [
            "image1.jpg",
            "image2.png",
            "image3.jpg",
        ]
        if 'image_index' not in st.session_state:
            st.session_state.image_index = 0
        countdown = st_autorefresh(interval=3000, key="auto_refresh1")
        def update_image_index():
            st.session_state.image_index += 1
            if st.session_state.image_index >= len(images):
                st.session_state.image_index = 0
        update_image_index()
        image_width = 10
        center_container = st.container()
        with center_container:
            st.image(images[st.session_state.image_index], width=image_width, use_column_width=True)

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: yellow;'>Modern Problems require , Modern Solution</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""##### - The trend of online shopping has grown significantly over the past few years, with more people opting to purchase items online.""")
        st.write("""##### - This shift includes categories that were traditionally bought in-store, such as furniture, fashion, and even fast food.""")
        st.write("""##### - This change is driven by various factors such as convenience, a wider range of options, and often better deals and discounts available online.""")
        st.write("""##### - The whole world loves online shopping. Payments made via UPI increased by 80% over the previous fiscal year.""")
        st.write("""##### - UPI accounted for more than 75% of the total transaction volume of India's retail digital payments in February 2023.""")
        st.write("""##### - In the first quarter of FY 2023–24, transaction volume reached 24.9 billion and transaction value reached INR 39.7 trillion.""")

    with col2:
        image = Image.open("image4.png")
        image = image.resize((600, 350))
        st.image(image)

    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: yellow;'>Key Features</h3>", unsafe_allow_html=True)
    st.write(" ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""##### 1. Real-time Fraud Detection""")
        st.write("Instantly analyze transactions to identify potential fraudulent activity.")
        image = Image.open("image1.jpg")
        st.image(image)
        st.write("""##### 3. User-Friendly Interface""")
        st.write("Simple and intuitive design that requires minimal training for users.")
        image = Image.open("image2.png")
        st.image(image)

    with col2:
        st.write("""##### 2. Advanced Machine Learning Model""")
        st.write("Leverage sophisticated algorithms trained on extensive historical data.")
        image = Image.open("image3.jpg")
        st.image(image)
        st.write("""##### 4. Prediction History Maintenance""")
        st.write("Maintain a detailed history of all predictions made, allowing for review and analysis of past fraud detection results.")
        image = Image.open("image4.png")
        st.image(image)

    # with col3:
    #     image = Image.open("image5.png")
    #     image = image.resize((600, 350))
    #     st.image(image)


if selected == "Single-Predict":

    def find_type(text):
        if text == "CASH_IN":
            return 0
        elif text == "CASH_OUT":
            return 1
        elif text == "DEBIT":
            return 2
        elif text == "PAYMENT":
            return 3
        else:
            return 4

    # Columns
    col1, col2 = st.columns(2)

    with col1:
        types = st.selectbox("Transaction Type", ("Select", "CASH_IN(0)", "CASH_OUT(1)", "DEBIT(2)", "PAYMENT(3)", "TRANSFER(4)"))
        oldbalanceOrg = st.number_input("Old Balance Original", min_value=0.0, format="%.2f")

    with col2:
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        newbalanceOrg = st.number_input("New Balance Original", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        if (types=="Select" or amount == 0.0):
            message = "⚠️ Please fill all the required fields."
            st.markdown(f"<p style='font-size:24px; color:red;'>Error: {message}</p>", unsafe_allow_html=True)
            # st.error("Please fill all the required fields.")
        else:
            types = find_type(types)
            test = np.array([[types, amount, oldbalanceOrg, newbalanceOrg]])
            res = model.predict(test)
            st.session_state.prediction_history.append({
                    "Amount": amount,
                    "Old Balance Origin": oldbalanceOrg,
                    "New Balance Origin": newbalanceOrg,
                    "Prediction": "Fraudulent" if res == 1 else "Not Fraudulent"
                })
            
            if res == 1:
                message = "⚠️ The transaction is predicted to be fraudulent."
                st.markdown(f"<p style='font-size:24px; color:red;'>Prediction: {message}</p>", unsafe_allow_html=True)
                # st.error("Prediction: " + "⚠️ The transaction is predicted to be fraudulent.")
                st.session_state.expander_open = True
            else:
                message = "✅ The transaction is predicted to be non-fraudulent."
                st.markdown(f"<p style='font-size:24px; color:green;'>Prediction: {message}</p>", unsafe_allow_html=True)
                # st.success("Prediction: " + "✅ The transaction is predicted to be non-fraudulent.")
                st.session_state.expander_open = False



if selected == "File-Predict":
    header = st.container()

    # Define columns and type dictionary
    columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    type_dict = {
        'PAYMENT': 0,
        'TRANSFER': 1,
        'CASH_OUT': 2, 
        'DEBIT': 3, 
        'CASH_IN': 4
    }

    def predict_batch(data):
        return model.predict(data)

    output = {0: 'Not Fraud', 1: 'Fraud'}

    with header:
        st.header("Predict Multiple Transactions")
        st.write("Upload CSV file to check multiple transactions.")
        st.write("Please follow the column structure as shown below:")
        st.write("1. type")
        st.write("2. amount")
        st.write("3. oldbalanceOrg")
        st.write("4. newbalanceOrig")

        
    with st.expander("Upload .csv file to predict your transactions", expanded=True):
        uploaded_file = st.file_uploader("Choose file:", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.write(f"Uploaded file contains {df.shape[0]} records and {df.shape[1]} attributes.")
        
            if st.button("Predict"):
                try:
                    # Prepare the data for prediction
                    new_df = df[columns].copy()
                    new_df['type'] = new_df['type'].replace(type_dict)

                    # Ensure numeric conversion
                    new_df[columns] = new_df[columns].apply(pd.to_numeric, errors='coerce')

                    # Predict
                    predictions = predict_batch(new_df)
                    df['isFraud'] = predictions
                    df['isFraud'] = df['isFraud'].replace(output)

                    # Display results
                    st.write("Predictions are succesfully stored in the file..")
                    
                    # Download the updated CSV file
                    st.download_button(label="Download updated CSV", data=df.to_csv(index=False), file_name='predicted_transactions.csv', mime='text/csv')
                except KeyError as e:
                    st.error(f"Key error: {e}. Please ensure the uploaded CSV contains the required columns.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if selected == "History":
    st.title("Prediction History")
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No predictions made yet.")



if selected == "About":

    st.markdown("<h3 style='text-align: center; color: yellow;'>How to Use?</h3>", unsafe_allow_html=True)
    st.write(" ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""##### 1. Select Transaction Type""")
        st.write("Choose the type of transaction (e.g., CASH_IN, CASH_OUT).")
        st.write("""##### 2. Enter Amount""")
        st.write("Input the transaction amount.")
        st.write("""##### 3. Enter Old Balance Original""")
        st.write("Provide the original balance before the transaction.")
        st.write("""##### 4. Enter New Balance Original""")
        st.write("Provide the new balance after the transaction.")
        st.write("""##### 5. Click 'Predict'""")
        st.write("Click the 'Predict' button to determine if the transaction is fraudulent.")
        st.write("""##### 6. View Results""")
        st.write("The prediction result will be displayed indicating whether the transaction is fraudulent or non-fraudulent.")

    with col2:
        st.write(" ")
        image = Image.open("image6.jpg")
        image = image.resize((500, 500))
        st.image(image)

    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: yellow;'>Random Forest Algorithm in Machine Learning</h3>", unsafe_allow_html=True)
    st.write(" ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""##### 1. Random Forest algorithm is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase.""")
        st.write("""##### 2. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition.""")
        st.write("""##### 3. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.""")
        st.write("""##### 4. Random forests are widely used for classification and regression functions, which are known for their ability to handle complex data, reduce overfitting, and provide reliable forecasts in different environments.""")
        st.write("""##### 5. Key Features of Random Forest:""")
        st.write("A. High Predictive Accuracy")
        st.write("B. Resistance to Overfitting")
        st.write("C. Large Datasets Handling")
    with col2:
        st.write(" ")
        st.write(" ")
        image = Image.open("image7.png")
        image = image.resize((700, 400))
        st.image(image)
