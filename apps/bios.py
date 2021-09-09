import streamlit as st
import matplotlib.pyplot as plt

def app():
    st.title('About Us')
    st.write('SERSA Team')

    team = {
        '01': 'Filipo',
        '02': 'Ia',
        '03': 'Michael',
        '04': 'Pankaj'
    }

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in team.keys():
        filename = f'bios/team{i}.jpg'
        fig.add_subplot(2, 2, int(i))
        plt.title(team[i])
        image = plt.imread(filename)
        plt.axis('off')
        plt.imshow(image)
    st.pyplot(fig)
