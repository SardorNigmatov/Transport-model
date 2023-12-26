import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader("Rasmni yuklash", type=['png', 'jpeg', 'svg','jfif'])

# Check if a file is uploaded
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('transport_model.pkl')

    # Bashorat
    pred, pred_id , probs  = model.predict(img)

    # Display prediction
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    #plot
    fig = px.bar(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)