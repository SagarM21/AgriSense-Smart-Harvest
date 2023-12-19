import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import shutil
import plotly.graph_objects as go
from PIL import Image, ImageOps
import numpy as np
import base64
import path
import json
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.models import load_model
import _pickle


# Load Model Old Approach

# # Load Crop Recommender model
# with open("./models/CropRecommender.pkl", "rb") as crop_recommender_file:
#     crop_recommender_model = pickle.load(crop_recommender_file)

# # Load Plant Disease Detection model
# plant_disease_model = load_model("./models/Plant_Disease.hdf5")


# Load Crop Recommender model
try:
    with open("./models/CropRecommender.pkl", "rb") as crop_recommender_pickle:
        crop_recommender_model = pickle.load(crop_recommender_pickle)
except _pickle.UnpicklingError as e:
    print(f"Error loading CropRecommender model: {e}")
    crop_recommender_model = None

# Load Plant Disease Detection model
plant_disease_model = load_model("./models/Plant_Disease.hdf5")

# Function to preprocess image and make predictions
def import_and_predict(image_data, model):
    img = ImageOps.fit(image_data, size=(220, 220))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255
    result = model.predict([np.expand_dims(x, axis=0)])
    return result

# Function to predict crop yield
def yield_prediction(input_list_yield):
    one_hot_encoder = {}
    with open("./models/OneHotEncoder.pkl", "rb") as ohe_pickle:
        one_hot_encoder = pickle.load(ohe_pickle)

    with open("./models/classifier.pkl", "rb") as classifier_pickle:
        model = pickle.load(classifier_pickle)

    with open("./models/list_mapping.pkl", "rb") as encodings_pickle:
        encodings = pickle.load(encodings_pickle)

    input_array_df = pd.DataFrame(input_list_yield).T
    columns_to_drop = [0, 1]
    one_hot_encoded_feature = one_hot_encoder.transform(input_array_df[columns_to_drop]).toarray()
    df_encoded = pd.DataFrame(one_hot_encoded_feature)
    df_final = pd.concat([df_encoded, input_array_df.drop(columns_to_drop, axis=1)], axis=1)

    X = df_final.values
    X[0, 680] = encodings[0][X[0, 680]]
    X[0, 681] = encodings[1][X[0, 681]]

    prediction = model.predict(X.reshape(1, -1))
    prediction = float(prediction)
    return prediction

# Function to convert image to base64 bytes
def img_to_bytes(img_path):
    img_bytes = path.Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Streamlit application
def main():
    # HTML template for styling
    html_template = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"><font size="10"> AgriSense Smart Harvest🌱 </font></h1>
    </div>
    <!-- Background styling -->
    <style>
    .reportview-container .main {
        background: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png");
        background-size: cover;
    }
   .sidebar .sidebar-content {
        background: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png")
    }
    </style>
    <style>
    body {
    background-image: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png");
    background-size: cover;
    }
    </style>
    """

    st.markdown(html_template, unsafe_allow_html=True)

    # UI for selecting functionalities
    selection = st.radio(
        "",
        [
            "Crop Disease Detection",
            "Crop Recommendation",
            "Yield Prediction",
        ],
    )

    # Streamlit styling
    st.write(
        """<style>
            .reportview-container .markdown-text-container {
                font-family: monospace;
            }
            .sidebar .sidebar-content {
                background-image: linear-gradient(#FFFFFF,#FFFFFF);
                color: white;
            }
            .Widget>label {
                color: white;
                font-family: monospace;
            }
            [class^="st-b"]  {
                color: white;
                font-family: monospace;
            }
            .st-bb {
                background-color: transparent;
            }
            .st-at {
                
            }
            footer {
                font-family: monospace;
            }
            .reportview-container .main footer, .reportview-container .main footer a {
                color: #FFFFFF;
            }
            header .decoration {
                background-image: none;
            }

            </style>""",
        unsafe_allow_html=True,
    )

    if selection == "Crop Disease Detection":
        # UI and logic for Crop Disease Detection
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Diseases Detection</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> In recent times, drastic climate changes and lack of immunity in crops has caused substantial increase in growth of crop diseases. This causes large scale demolition of crops, decreases cultivation and eventually leads to financial loss of farmers. Due to rapid growth in variety of diseases , identification and treatment of the disease is a major importance.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        file = st.sidebar.file_uploader("Please upload a crop image")

        if st.button("Detect"):
            if file is None:
                st.sidebar.text("please upload an image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict(image, plant_disease_model)
                file_json = open("./models/class_indices.json", "r")
                class_indices = json.load(file_json)
                classes = list(class_indices.keys())
                classresult = np.argmax(predictions, axis=1)
                word = classes[classresult[0]].split("__")
                word[0] = word[0].replace("_", " ")
                word[1] = word[1].replace("_", " ")
                st.success("This crop is {} and it has {} ".format(
                    word[0], word[1]))

    elif selection == "Crop Recommendation":
        # UI and logic for Crop Recommendation
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Recommendation</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown(
            " Find out the most suitable crop to grow in your farm 👨‍🌾")
        nitrogen = st.sidebar.number_input("Nitrogen", 1, 10000)
        phosphorus = st.sidebar.number_input("Phosporus", 1, 10000)
        potassium = st.sidebar.number_input("Potassium", 1, 10000)
        temperature = st.sidebar.number_input("Temperature", 0.0, 100000.0)
        humidity = st.sidebar.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.sidebar.number_input("Ph", 0.0, 100000.0)
        rainfall = st.sidebar.number_input("Rainfall in mm", 0.0, 100000.0)

        features_list = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        single_prediction = np.array(features_list).reshape(1, -1)

        if st.button("Predict"):
            prediction = crop_recommender_model.predict(single_prediction)
            st.success(
                f"{prediction.item().title()} are recommended by the A.I for your farm."
            )
            
    elif selection == "Yield Prediction":
        # UI and logic for Crop Yield Prediction
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Yield Prediction</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b>Forecasting or predicting the crop yield well ahead of its harvest time would assist the strategists and farmers for taking suitable measures for selling and storage. In addition to such human errors, the fluctuations in the prices themselves make creating a stable and robust forecasting solution a necessity.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        state_selection = st.sidebar.selectbox(
            "Select Your State",
            ("Andaman and Nicobar Islands","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chandigarh","Chhattisgarh","Dadra and Nagar Haveli","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir ","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Puducherry","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana ","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
            ),
        )

        district_selection = st.sidebar.selectbox(
            "Select Your District",
            ("NICOBARS","NORTH AND MIDDLE ANDAMAN","SOUTH ANDAMANS","ANANTAPUR","CHITTOOR","EAST GODAVARI","GUNTUR","KADAPA","KRISHNA","KURNOOL","PRAKASAM","SPSR NELLORE","SRIKAKULAM","VISAKHAPATANAM","VIZIANAGARAM","WEST GODAVARI","ANJAW","CHANGLANG","DIBANG VALLEY","EAST KAMENG","EAST SIANG","KURUNG KUMEY","LOHIT","LONGDING","LOWER DIBANG VALLEY","LOWER SUBANSIRI","NAMSAI","PAPUM PARE","TAWANG","TIRAP","UPPER SIANG","UPPER SUBANSIRI","WEST KAMENG","WEST SIANG","BAKSA","BARPETA","BONGAIGAON","CACHAR","CHIRANG","DARRANG","DHEMAJI","DHUBRI","DIBRUGARH","DIMA HASAO","GOALPARA","GOLAGHAT","HAILAKANDI","JORHAT","KAMRUP","KAMRUP METRO","KARBI ANGLONG","KARIMGANJ","KOKRAJHAR","LAKHIMPUR","MARIGAON","NAGAON","NALBARI","SIVASAGAR","SONITPUR","TINSUKIA","UDALGURI","ARARIA","ARWAL","AURANGABAD","BANKA","BEGUSARAI","BHAGALPUR","BHOJPUR","BUXAR","DARBHANGA","GAYA","GOPALGANJ","JAMUI","JEHANABAD","KAIMUR (BHABUA)","KATIHAR","KHAGARIA","KISHANGANJ","LAKHISARAI","MADHEPURA","MADHUBANI","MUNGER","MUZAFFARPUR","NALANDA","NAWADA","PASHCHIM CHAMPARAN","PATNA","PURBI CHAMPARAN","PURNIA","ROHTAS","SAHARSA","SAMASTIPUR","SARAN","SHEIKHPURA","SHEOHAR","SITAMARHI","SIWAN","SUPAUL","VAISHALI","CHANDIGARH","BALOD","BALODA BAZAR","BALRAMPUR","BASTAR","BEMETARA","BIJAPUR","BILASPUR","DANTEWADA","DHAMTARI","DURG","GARIYABAND","JANJGIR-CHAMPA","JASHPUR","KABIRDHAM","KANKER","KONDAGAON","KORBA","KOREA","MAHASAMUND","MUNGELI","NARAYANPUR","RAIGARH","RAIPUR","RAJNANDGAON","SUKMA","SURAJPUR","SURGUJA","DADRA AND NAGAR HAVELI","NORTH GOA","SOUTH GOA","AHMADABAD","AMRELI","ANAND","BANAS KANTHA","BHARUCH","BHAVNAGAR","DANG","DOHAD","GANDHINAGAR","JAMNAGAR","JUNAGADH","KACHCHH","KHEDA","MAHESANA","NARMADA","NAVSARI","PANCH MAHALS","PATAN","PORBANDAR","RAJKOT","SABAR KANTHA","SURAT","SURENDRANAGAR","TAPI","VADODARA","VALSAD","AMBALA","BHIWANI","FARIDABAD","FATEHABAD","GURGAON","HISAR","JHAJJAR","JIND","KAITHAL","KARNAL","KURUKSHETRA","MAHENDRAGARH","MEWAT","PALWAL","PANCHKULA","PANIPAT","REWARI","ROHTAK","SIRSA","SONIPAT","YAMUNANAGAR","CHAMBA","HAMIRPUR","KANGRA","KINNAUR","KULLU","LAHUL AND SPITI","MANDI","SHIMLA","SIRMAUR","SOLAN","UNA","ANANTNAG","BADGAM","BANDIPORA","BARAMULLA","DODA","GANDERBAL","JAMMU","KARGIL","KATHUA","KISHTWAR","KULGAM","KUPWARA","LEH LADAKH","POONCH","PULWAMA","RAJAURI","RAMBAN","REASI","SAMBA","SHOPIAN","SRINAGAR","UDHAMPUR","BOKARO","CHATRA","DEOGHAR","DHANBAD","DUMKA","EAST SINGHBUM","GARHWA","GIRIDIH","GODDA","GUMLA","HAZARIBAGH","JAMTARA","KHUNTI","KODERMA","LATEHAR","LOHARDAGA","PAKUR","PALAMU","RAMGARH","RANCHI","SAHEBGANJ","SARAIKELA KHARSAWAN","SIMDEGA","WEST SINGHBHUM","BAGALKOT","BANGALORE RURAL","BELGAUM","BELLARY","BENGALURU URBAN","BIDAR","CHAMARAJANAGAR","CHIKBALLAPUR","CHIKMAGALUR","CHITRADURGA","DAKSHIN KANNAD","DAVANGERE","DHARWAD","GADAG","GULBARGA","HASSAN","HAVERI","KODAGU","KOLAR","KOPPAL","MANDYA","MYSORE","RAICHUR","RAMANAGARA","SHIMOGA","TUMKUR","UDUPI","UTTAR KANNAD","YADGIR","ALAPPUZHA","ERNAKULAM","IDUKKI","KANNUR","KASARAGOD","KOLLAM","KOTTAYAM","KOZHIKODE","MALAPPURAM","PALAKKAD","PATHANAMTHITTA","THIRUVANANTHAPURAM","THRISSUR","WAYANAD","AGAR MALWA","ALIRAJPUR","ANUPPUR","ASHOKNAGAR","BALAGHAT","BARWANI","BETUL","BHIND","BHOPAL","BURHANPUR","CHHATARPUR","CHHINDWARA","DAMOH","DATIA","DEWAS","DHAR","DINDORI","GUNA","GWALIOR","HARDA","HOSHANGABAD","INDORE","JABALPUR","JHABUA","KATNI","KHANDWA","KHARGONE","MANDLA","MANDSAUR","MORENA","NARSINGHPUR","NEEMUCH","PANNA","RAISEN","RAJGARH","RATLAM","REWA","SAGAR","SATNA","SEHORE","SEONI","SHAHDOL","SHAJAPUR","SHEOPUR","SHIVPURI","SIDHI","SINGRAULI","TIKAMGARH","UJJAIN","UMARIA","VIDISHA","AHMEDNAGAR","AKOLA","AMRAVATI","BEED","BHANDARA","BULDHANA","CHANDRAPUR","DHULE","GADCHIROLI","GONDIA","HINGOLI","JALGAON","JALNA","KOLHAPUR","LATUR","MUMBAI","NAGPUR","NANDED","NANDURBAR","NASHIK","OSMANABAD","PALGHAR","PARBHANI","PUNE","RAIGAD","RATNAGIRI","SANGLI","SATARA","SINDHUDURG","SOLAPUR","THANE","WARDHA","WASHIM","YAVATMAL","BISHNUPUR","CHANDEL","CHURACHANDPUR","IMPHAL EAST","IMPHAL WEST","SENAPATI","TAMENGLONG","THOUBAL","UKHRUL","EAST GARO HILLS","EAST JAINTIA HILLS","EAST KHASI HILLS","NORTH GARO HILLS","RI BHOI","SOUTH GARO HILLS","SOUTH WEST GARO HILLS","SOUTH WEST KHASI HILLS","WEST GARO HILLS","WEST JAINTIA HILLS","WEST KHASI HILLS","AIZAWL","CHAMPHAI","KOLASIB","LAWNGTLAI","LUNGLEI","MAMIT","SAIHA","SERCHHIP","DIMAPUR","KIPHIRE","KOHIMA","LONGLENG","MOKOKCHUNG","MON","PEREN","PHEK","TUENSANG","WOKHA","ZUNHEBOTO","ANUGUL","BALANGIR","BALESHWAR","BARGARH","BHADRAK","BOUDH","CUTTACK","DEOGARH","DHENKANAL","GAJAPATI","GANJAM","JAGATSINGHAPUR","JAJAPUR","JHARSUGUDA","KALAHANDI","KANDHAMAL","KENDRAPARA","KENDUJHAR","KHORDHA","KORAPUT","MALKANGIRI","MAYURBHANJ","NABARANGPUR","NAYAGARH","NUAPADA","PURI","RAYAGADA","SAMBALPUR","SONEPUR","SUNDARGARH","KARAIKAL","MAHE","PONDICHERRY","YANAM","AMRITSAR","BARNALA","BATHINDA","FARIDKOT","FATEHGARH SAHIB","FAZILKA","FIROZEPUR","GURDASPUR","HOSHIARPUR","JALANDHAR","KAPURTHALA","LUDHIANA","MANSA","MOGA","MUKTSAR","NAWANSHAHR","PATHANKOT","PATIALA","RUPNAGAR","S.A.S NAGAR","SANGRUR","TARN TARAN","AJMER","ALWAR","BANSWARA","BARAN","BARMER","BHARATPUR","BHILWARA","BIKANER","BUNDI","CHITTORGARH","CHURU","DAUSA","DHOLPUR","DUNGARPUR","GANGANAGAR","HANUMANGARH","JAIPUR","JAISALMER","JALORE","JHALAWAR","JHUNJHUNU","JODHPUR","KARAULI","KOTA","NAGAUR","PALI","PRATAPGARH","RAJSAMAND","SAWAI MADHOPUR","SIKAR","SIROHI","TONK","UDAIPUR","EAST DISTRICT","NORTH DISTRICT","SOUTH DISTRICT","WEST DISTRICT","ARIYALUR","COIMBATORE","CUDDALORE","DHARMAPURI","DINDIGUL","ERODE","KANCHIPURAM","KANNIYAKUMARI","KARUR","KRISHNAGIRI","MADURAI","NAGAPATTINAM","NAMAKKAL","PERAMBALUR","PUDUKKOTTAI","RAMANATHAPURAM","SALEM","SIVAGANGA","THANJAVUR","THE NILGIRIS","THENI","THIRUVALLUR","THIRUVARUR","TIRUCHIRAPPALLI","TIRUNELVELI","TIRUPPUR","TIRUVANNAMALAI","TUTICORIN","VELLORE","VILLUPURAM","VIRUDHUNAGAR","ADILABAD","HYDERABAD","KARIMNAGAR","KHAMMAM","MAHBUBNAGAR","MEDAK","NALGONDA","NIZAMABAD","RANGAREDDI","WARANGAL","DHALAI","GOMATI","KHOWAI","NORTH TRIPURA","SEPAHIJALA","SOUTH TRIPURA","UNAKOTI","WEST TRIPURA","AGRA","ALIGARH","ALLAHABAD","AMBEDKAR NAGAR","AMETHI","AMROHA","AURAIYA","AZAMGARH","BAGHPAT","BAHRAICH","BALLIA","BANDA","BARABANKI","BAREILLY","BASTI","BIJNOR","BUDAUN","BULANDSHAHR","CHANDAULI","CHITRAKOOT","DEORIA","ETAH","ETAWAH","FAIZABAD","FARRUKHABAD","FATEHPUR","FIROZABAD","GAUTAM BUDDHA NAGAR","GHAZIABAD","GHAZIPUR","GONDA","GORAKHPUR","HAPUR","HARDOI","HATHRAS","JALAUN","JAUNPUR","JHANSI","KANNAUJ","KANPUR DEHAT","KANPUR NAGAR","KASGANJ","KAUSHAMBI","KHERI","KUSHI NAGAR","LALITPUR","LUCKNOW","MAHARAJGANJ","MAHOBA","MAINPURI","MATHURA","MAU","MEERUT","MIRZAPUR","MORADABAD","MUZAFFARNAGAR","PILIBHIT","RAE BARELI","RAMPUR","SAHARANPUR","SAMBHAL","SANT KABEER NAGAR","SANT RAVIDAS NAGAR","SHAHJAHANPUR","SHAMLI","SHRAVASTI","SIDDHARTH NAGAR","SITAPUR","SONBHADRA","SULTANPUR","UNNAO","VARANASI","ALMORA","BAGESHWAR","CHAMOLI","CHAMPAWAT","DEHRADUN","HARIDWAR","NAINITAL","PAURI GARHWAL","PITHORAGARH","RUDRA PRAYAG","TEHRI GARHWAL","UDAM SINGH NAGAR","UTTAR KASHI","24 PARAGANAS NORTH","24 PARAGANAS SOUTH","BANKURA","BARDHAMAN","BIRBHUM","COOCHBEHAR","DARJEELING","DINAJPUR DAKSHIN","DINAJPUR UTTAR","HOOGHLY","HOWRAH","JALPAIGURI","MALDAH","MEDINIPUR EAST","MEDINIPUR WEST","MURSHIDABAD","NADIA","PURULIA",
            ),
        )

        crop_year = 2014

        season_select = st.sidebar.selectbox(
            "Select The Season",
            ("Kharif", "Whole Year", "Autumn", "Rabi", "Summer", "Winter"),
        )

        # Mapping season names
        season_mapping = {
            "Kharif": "Kharif     ",
            "Whole Year": "Whole Year ",
            "Autumn": "Autumn     ",
            "Rabi": "Rabi       ",
            "Summer": "Summer     ",
            "Winter": "Winter     "
        }
        season_select = season_mapping.get(season_select, "Winter")

        crop_selection = st.sidebar.selectbox(
            "Select Your Crop",
            ("Arecanut","Other Kharif pulses","Rice","Banana","Cashewnut","Coconut ","Dry ginger","Sugarcane","Sweet potato","Tapioca","Black pepper","Dry chillies","other oilseeds","Turmeric","Maize","Moong(Green Gram)","Urad","Arhar/Tur","Groundnut","Sunflower","Bajra","Castor seed","Cotton(lint)","Horse-gram","Jowar","Korra","Ragi","Tobacco","Gram","Wheat","Masoor","Sesamum","Linseed","Safflower","Onion","other misc. pulses","Samai","Small millets","Coriander","Potato","Other  Rabi pulses","Soyabean","Beans & Mutter(Vegetable)","Bhindi","Brinjal","Citrus Fruit","Cucumber","Grapes","Mango","Orange","other fibres","Other Fresh Fruits","Other Vegetables","Papaya","Pome Fruit","Tomato","Rapeseed &Mustard","Mesta","Cowpea(Lobia)","Lemon","Pome Granet","Sapota","Cabbage","Peas  (vegetable)","Niger seed","Bottle Gourd","Sannhamp","Varagu","Garlic","Ginger","Oilseeds total","Pulses total","Jute","Peas & beans (Pulses)","Blackgram","Paddy","Pineapple","Barley","Khesari","Guar seed","Moth","Other Cereals & Millets","Cond-spcs other","Turnip","Carrot","Redish","Arcanut (Processed)","Atcanut (Raw)","Cashewnut Processed","Cashewnut Raw","Cardamom","Rubber","Bitter Gourd","Drum Stick","Jack Fruit","Snak Guard","Pump Kin","Tea","Coffee","Cauliflower","Other Citrus Fruit","Water Melon","Total foodgrain","Kapas","Colocosia","Lentil","Bean","Jobster","Perilla","Rajmash Kholar","Ricebean (nagadal)","Ash Gourd","Beet Root","Lab-Lab","Ribed Guard","Yam","Apple","Peach","Pear","Plums","Litchi","Ber","Other Dry Fruit","Jute & mesta",
            ),
        )

        area_selection = st.sidebar.text_input(
            "Enter the Area of Your Field (in square meters)"
        )

        result = ""
        if st.button("Predict"):
            result = yield_prediction(
                [
                    state_selection,
                    district_selection,
                    crop_year,
                    season_select,
                    crop_selection,
                    area_selection,
                ]
            )

        st.success(
            "The estimated Crop Production (Kg per hectare) is {}".format(
                result)
        )

    # Hide Streamlit style
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Footer with team information
    st.markdown(
        """<style>footer {visibility: hidden;} footer:after {content:'Capstone Project By Team 21 💥';visibility: visible;display: block;position: relative;#background-color: red;padding: 5px; top: 2px;}</style>""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
