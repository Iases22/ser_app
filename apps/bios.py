import streamlit as st
import matplotlib.pyplot as plt
import base64


def app():

    # main page and sidebar background images
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

    # fig, ax = plt.subplots()
    # #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # for i in team.keys():
    #     filename = f'bios/team{i}.jpg'
    #     plt.subplot(2, 2, int(i))
    #     #plt.title(team[i])
    #     ax.set_title(label=team[i], fontsize=5)
    #     image = plt.imread(filename)
    #     plt.axis('off')
    #     plt.imshow(image)
    # st.pyplot(fig)

    fig = plt.subplots()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in team.keys():
        filename = f'bios/team{i}.jpg'
        fig.add_subplot(2, 2, int(i))
        plt.title(team[i], fontsize=5)
        image = plt.imread(filename)
        plt.axis('off')
        plt.imshow(image)
    st.pyplot(fig)

# fig, ax = plt.subplots(figsize=(8, 1))
#             right_side = ax.spines["right"]
#             top_side = ax.spines['top']
#             right_side.set_visible(False)
#             top_side.set_visible(False)

#             ax.barh(reverse_ranked_emotions,
#                     reverse_ranked_values,
#                     color=['r', 'y', 'g', 'b', 'c', 'm'])

#             ax.set_yticklabels(reverse_ranked_emotions, fontsize=5)
#             ax.set_xticklabels(list(range(0, 100, 10)), fontsize=5)

#             for index, value in enumerate(reverse_ranked_values):
#                 if value < 0.1:
#                     continue
#                 plt.text(value, index, str(value), fontsize=5)

#             st.pyplot(fig)
