import streamlit as st
import matplotlib.pyplot as plt
import base64


def app():

    # page and sidebar background images
    main_bg = "background5.jpg"
    main_bg_ext = "jpg"
    side_bg = "background4.jpg"
    side_bg_ext = "jpg"
    st.markdown(f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        .css-17eq0hr {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
        }}
        </style>
        """,
                unsafe_allow_html=True)

    st.title('About Us')
    st.write('SERSA Team')

    team = {
        '01': 'Filippo Colonna',
        '02': 'Ia Vaquilar',
        '03': 'Michael Michael',
        '04': 'Pankaj Patel'
    }

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in team.keys():
        filename = f'bios/team{i}.jpg'
        fig.add_subplot(2, 2, int(i))
        plt.title(team[i], fontsize=6)
        image = plt.imread(filename)
        plt.axis('off')
        plt.imshow(image)
    st.pyplot(fig)
