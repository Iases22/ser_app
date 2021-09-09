import streamlit as st
from pages import MultiApp
from apps import sersa, features, bios

app = MultiApp()

#add pages here
app.add_app("SERSA", sersa.app)
app.add_app("Sound Features", features.app)
app.add_app("About us", bios.app)

# The main app
app.run()
