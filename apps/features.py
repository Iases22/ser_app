# import streamlit as st
# import matplotlib.pyplot as plt
# from ser_app.spectrogram import create_spec
# import base64

# def app():

#     # main page and sidebar background images
#     main_bg = "background5.jpg"
#     main_bg_ext = "jpg"
#     side_bg = "background4.jpg"
#     side_bg_ext = "jpg"
#     st.markdown(f"""
#         <style>
#         .reportview-container {{
#             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#         }}
#         .css-17eq0hr {{
#             background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
#         }}
#         </style>
#         """,
#                 unsafe_allow_html=True)

#     st.title('Sound Features')
#     st.write('Features used in the project in spectrogram:')
#     spec = create_spec()
#     st.pyplot(spec)
