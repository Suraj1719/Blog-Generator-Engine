#for GUI
import streamlit as st
#pre_defined recipes for generating prompts for language models
from langchain.prompts import PromptTemplate
#model that can generate text using GPU GPTQ
#use this model to create various types of content, such as stories, poems, code etc.
from langchain.llms import CTransformers
#to track tie span of program
import time

# link to models : - https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

def get_llam_response(input_text,no_words,blog_style,model_size):
    #loading models
    m1_4GB="llama-2-7b-chat.ggmlv3.q4_1.bin"
    m2_2GB="llama-2-7b-chat.ggmlv3.q2_k.bin"

    model_chosen=m1_4GB if model_size=="4.33GB" else m2_2GB
    
    model=CTransformers(model=model_chosen,
                        model_type='llama',
                        config={
                            'max_new_tokens':256,
                            'temperature':0.01
                        })
    
    # prompt Template
    template="""
    write a blog post on topic {input_text} assuming the reader is {blog_style} within 
    {no_words} words.
    """

    #creating prompt
    prompt =PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                           template=template)
    
    #generating response from llama2 model
    response =model(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response





st.set_page_config(
    page_title="Blog Generator",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Blog Generator")
# taking blog titla from user
title=st.text_input("Enter blog title")

#dividing page in two parts
col1,col2,col3=st.columns([5,5,5])

with col1:
    word_count=st.text_input("Word Limit")
with col2:
    audience= st.selectbox("Target Audience",
                           ('Researcher','Data Scientist','Common people'),
                           index=0
                           )
with col3:
    model_size=st.selectbox("Choose model size",
                            ('2.5GB model','4.33GB model'),
                            index=0
                            )
    
submit = st.button("Generate")

if submit:
    start_time=time.time()
    st.write(get_llam_response(title,word_count,audience,model_size))
    end_time=time.time()

    elapsed_time=end_time-start_time
    print(f"Time Taken: {elapsed_time} seconds")
    st.write(f"Time Taken: {elapsed_time} seconds")