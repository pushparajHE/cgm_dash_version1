import CGM_DASH
import streamlit as st
import movement_dash
import base64


with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
    
PAGES = {
    "Metabolic Index": CGM_DASH,
    "Cardiovascular Index":movement_dash}
#st.snow()
st.sidebar.image("HE_Word.png", width=300)
st.sidebar.title('Navigation')

selection = st.sidebar.radio("Go to Index", list(PAGES.keys()))
page = PAGES[selection]
page.app()





def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.sidebar.markdown(page_bg_img, unsafe_allow_html=True)
    
#set_background("HE_BACK.png")
