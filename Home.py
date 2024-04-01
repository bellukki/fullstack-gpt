import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🌌"
)

with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")

st.title("FullstackGPT Home")

st.markdown(
    """
Here are the apps I made:

- [x] [DocumentGPT](/DocumentGPT)
- [ ] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
    """
)
