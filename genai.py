import os
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from ZeroShotAnalyzeSettings import zero_shot_analyze_settings
from langchain.chains import RetrievalQA
import openai
from logging import exception
import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from tabulate import tabulate
from PIL import Image
from io import StringIO
from FewShotSettings import few_shot_settings


class few_shot_prompt_utility:

    def __init__(self, examples, prefix, suffix, input_variables, example_template, example_variables):
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix
        self.input_variables = input_variables
        self.example_template = example_template
        self.example_variables = example_variables

    def get_prompt_template(self):
        example_prompt = PromptTemplate(
            input_variables=self.example_variables,
            template=self.example_template
        )
        return example_prompt
    
    def get_embeddings(self):
        embeddings = OpenAIEmbeddings(openai_api_key= st.secrets["openai_key"])
        return embeddings
    
    def get_example_selector(self, embeddings):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            embeddings,
            FAISS,
            k=3
        )
        return example_selector
        
    def get_prompt(self, question, example_selector, example_prompt):
        prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables
        )
        prompt = prompt_template.format(question=question, context="Inventory")
        return prompt_template
        

class zero_shot_analyze_utility:

    def __init__(self, question, ask, context, metadata):
        self.question = question
        self.ask = ask
        self.context = context
        self.metadata = metadata

    def get_analyze_prompt(self):
        template, variables = zero_shot_analyze_settings.get_prompt_template(self.ask, self.metadata)
        prompt_template = PromptTemplate(template=template, input_variables=variables)
        prompt_template.format(question=self.question, context=self.context)
        return prompt_template
        
        
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    max_tokens=2000,
    openai_api_key= st.secrets["openai_key"]
)

prefix = few_shot_settings.get_prefix()
suffix, input_variable = few_shot_settings.get_suffix()
examples = few_shot_settings.get_examples()
example_template, example_variables = few_shot_settings.get_example_template()
fewShot = few_shot_prompt_utility(examples=examples,
                                        prefix=prefix,
                                        suffix=suffix,
                                        input_variables=input_variable,
                                        example_template=example_template,
                                        example_variables=example_variables
                                        )


def fs_chain(question):
  """
  returns a question answer chain for faiss vectordb
  """
  example_prompt = fewShot.get_prompt_template()
  embeddings = fewShot.get_embeddings()
  example_selector = fewShot.get_example_selector(embeddings)
  prompt_template = fewShot.get_prompt(question, example_selector, example_prompt)
  docsearch = FAISS.load_local("faiss_index", embeddings)
  qa_chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": prompt_template})
  return qa_chain({"query": question})
  
def fs_analysis(dataframe,question):
  """
  returns a question answer chain for faiss vectordb
  """
  analysis_question_part1 ='''Provide analysis of the data in tabular format below. \n '''

  analysis_question_prompt = '''\nUse "Ask" and "Metadata" information as supporting data for the analysis. This information is mentioned toward end of this text.
        Keep analysis strictly for business users working in the inventory management domain to understand nature of output. Limit your response accordingly.
        Few Rules to follow are:
        1. If the result for the query is in tabular format make sure the whole analysis is in same format.
        2. The analysis must be within 80-100 words. 
        3. Do not include supplied data into analysis.
        '''
  
  analysis_question = analysis_question_part1 + dataframe  + analysis_question_prompt
  embeddings = fewShot.get_embeddings()
  docsearch = FAISS.load_local("faiss_index", embeddings)
  
  docs = docsearch.similarity_search(question)
  metadata = ""
  for i in docs:
    metadata = metadata + "\n" + i.page_content
  
  zeroShotAnlyze = zero_shot_analyze_utility(analysis_question, question, "inventory management", metadata)
  analyze_prompt = zeroShotAnlyze.get_analyze_prompt()
  qa_chain1 = RetrievalQA.from_chain_type(llm,
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": analyze_prompt})

  result = qa_chain1({"query": analysis_question})['result']
 
  return result

def output_operation(query_result,str_input):
    if len(query_result) >= 1:
        with st.chat_message("assistant"):
            df_2 = pd.DataFrame(query_result)
            df_analysis = str(df_2)
            analysis = fs_analysis(df_analysis,str_input)                    
            headers = df_2.columns
            st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{analysis}</p>', unsafe_allow_html=True)
            with st.expander("Table Output:"):
               st.markdown(tabulate(df_2, tablefmt="html",headers=headers,showindex=False), unsafe_allow_html = True) 
            #st.markdown(analysis)
            st.text("")
            with st.expander("The SQL query used for above question is:"):
              st.write(output['result'])
        data = df_2.to_csv(sep=',', index=False) + "<separator>" + analysis
        st.session_state.messages.append({"role": "assistant", "content": data})
    else:
        with st.chat_message("assistant"):
            err_msg = "Data for the provided question is not available. Please try to improve your question."
            st.markdown(err_msg)
            with st.expander("The SQL query used for above question is:"):
              st.write(output['result'])
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
    
st.set_page_config(layout="wide")

username=st.secrets["streamlit_username"]
password=st.secrets["streamlit_password"]
column_list = ["AMOUNT"]

# establish snowpark connection
conn = st.connection("snowpark")

# Reset the connection before using it if it isn't healthy
try:
    query_test = conn.query('select 1')
except:
    conn.reset()

# adding this to test out caching
st.cache_data(ttl=86400)
def sf_query(str_input):
    """
    performs snowflake query with caching
    """
    data = conn.query(str_input)
    return data

def creds_entered():
    if len(st.session_state["streamlit_username"])>0 and len(st.session_state["streamlit_password"])>0:
          if  st.session_state["streamlit_username"].strip() != username or st.session_state["streamlit_password"].strip() != password: 
              st.session_state["authenticated"] = False
              st.error("Invalid Username/Password ")

          elif st.session_state["streamlit_username"].strip() == username and st.session_state["streamlit_password"].strip() == password:
              st.session_state["authenticated"] = True


def authenticate_user():
      if "authenticated" not in st.session_state:
        buff, col, buff2 = st.columns([1,1,1])
        col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered) 
        col.text_input(label="Password", value="", key="streamlit_password", type="password", on_change=creds_entered)
        return False
      else:
           if st.session_state["authenticated"]: 
                return True
           else:  
                  buff, col, buff2 = st.columns([1,1,1])
                  col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered) 
                  col.text_input(label="Password:", value="", key="streamlit_password", type="password", on_change=creds_entered)
                  return False

if authenticate_user():
    with st.sidebar:
      image = Image.open("assets/jadeglobal.png")
      image = st.image('assets/jadeglobal.png',width=280)
      #st.markdown(""" ### SCM Inventory Management """)
      st.markdown("<h3 style='text-align: center;'>SCM Inventory Management</h3>", unsafe_allow_html=True)
      st.markdown("<h3 style='text-align: center;'>GenAI Assistant</h3>", unsafe_allow_html=True)
      
      
    str_input = st.chat_input("Enter your question:")
    st.markdown("""
    #### This is a demo to query inventory data from Snowflake for Marvell.
    #### Post your question in the text box below. 
              
    """)
    
    if "messages" not in st.session_state.keys():
          st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            role = message["role"]
            df_str = message["content"]
            if role == "user":
                st.markdown(df_str, unsafe_allow_html = True)
                continue
            if df_str.find("<separator>") > -1:
               csv_str = df_str[:df_str.index("<separator>")]
               analysis_str = df_str[df_str.index("<separator>") + len("<separator>"):]
               csv = StringIO(csv_str)
               df_data = pd.read_csv(csv, sep=',')
               df_data.columns = df_data.columns.str.replace('_', ' ')
               headers = df_data.columns
               st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{analysis_str}</p>', unsafe_allow_html=True)
               with st.expander("Table Output:"):
                  st.markdown(tabulate(df_data, tablefmt="html",headers=headers,showindex=False), unsafe_allow_html = True) 
               #st.markdown(analysis_str)
            else:
                st.markdown(df_str)
                
    if prompt := str_input:
        st.chat_message("user").markdown(prompt, unsafe_allow_html = True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            output = fs_chain(str_input)
            #st.write(output['result'])
            try:
                # if the output doesn't work we will try one additional attempt to fix it
                query_result = sf_query(output['result'])
                output_operation(query_result,str_input)
  
            except Exception as error:    
                  output = fs_chain(f'You need to fix the code but ONLY produce SQL code output. If the question is complex, consider using one or more CTE. Examine the DDL statements and answer this question: {output}')
                  query_result = sf_query(output['result'])
                  output_operation(query_result,str_input)
        except Exception as error:
          with st.chat_message("assistant"):
            err_msg = "Data for the provided question is not available."
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
