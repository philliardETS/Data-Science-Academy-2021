# PAUL HILLIARD - CAPSTONE PROJECT
# DATA SCIENCE ACADEMY
# 2021 - 2022

#import necessary libraries

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import json
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objs as go
pio.templates.default="ggplot2"



st.title("Data Science Academy 2021-2022")
st.markdown("## Capstone Project")
st.markdown("## Paul Hilliard")

st.markdown("---")
st.markdown("## Description of Data")
st.markdown("The datafile consists of a 20-item assessment, with item level timing and score performance.  Total assessment timing and score are also provided.  ")

st.markdown("Categorical record variables include Gender, Age, whether the assessment was taken on their Home Computer, and their Location." )

st.markdown("For my analysis, the Sum Score was grouped into Low, Mid and High performance.")
st.markdown("The Age variable was grouped into 5 equally-sized age categories.")

st.markdown("---")

	
#load in the science data using pandas
df = pd.read_csv("data/data_capstone_dsa2021_2022.csv")
df=pd.DataFrame(df)

# sum_score, gender, home_computer are clean.  
#bin the Ages and the Performance
df['age_group'] = pd.qcut(df['age'], 5, 
    labels=['Age Group1: < 28', 'Age Group2: 28-31', 'Age Group3: 32-36', 'Age Group4: 37-44', 'Age Group5: > 45'])
df['ability'] = pd.qcut(df['sum_score'], 3, labels=['Low', 'Mid', 'High'])

#recode the ability groups to use in a toggle bar
#recode the gender to 1=Males, 2=Females, to use in scatterplots
#and home_computer to 1=YES, 2=NO, to use in scatterplots

df.loc[df['sum_score'] <= 17, 'ability_group'] = 1
df.loc[df['sum_score'] == 18, 'ability_group'] = 2
df.loc[df['sum_score'] == 19, 'ability_group'] = 2
df.loc[df['sum_score'] == 20, 'ability_group'] = 3
	
df.loc[df['gender'] == 'Female', 'gender_coded'] = 1
df.loc[df['gender'] == 'Male', 'gender_coded'] = 2
df.loc[df['home_computer'] == 'No', 'home_computer_coded'] = 1
df.loc[df['home_computer'] == 'Yes', 'home_computer_coded'] = 2

st.title("The Data")

#take a look at the file
df
#df.tail(20)

#group the gender, age_group, and home_computer variables so they can be displayed in pie charts

gendercounts=pd.DataFrame(df.groupby('gender').count())
agecounts=pd.DataFrame(df.groupby('age_group').count())
homecompcounts=pd.DataFrame(df.groupby('home_computer').count())
genderlevs=["Female","Male"]
agelevs=["Age Group1: < 28", "Age Group2: 28-31", "Age Group3: 32-36", "Age Group4: 37-44", "Age Group5: > 45"]
homecomplevs=["Home Computer: No","Home Computer: Yes"]

#Pie Chart of some things
#1-3

st.title("Some Basic Categorical Displays")
st.markdown("## Pie Chart - Proportion of Records - by Gender")
fig=px.pie(gendercounts, values='sum_score', names=genderlevs)
fig.update_layout(uniformtext_minsize=35, uniformtext_mode='hide', legend_font_size=35)
st.plotly_chart(fig)
st.markdown("## Pie Chart - Proportion of Records - by Age Split into 5 Approximately Equal Groups")
fig=px.pie(agecounts, values='sum_score', names=agelevs)
fig.update_layout(uniformtext_minsize=30, uniformtext_mode='hide', legend_font_size=22)
st.plotly_chart(fig)
st.markdown("## Pie Chart - Proportion of Records - by Home Computer Use")
fig=px.pie(homecompcounts, values='sum_score', names=homecomplevs)
fig.update_layout(uniformtext_minsize=35, uniformtext_mode='hide', legend_font_size=22)
st.plotly_chart(fig)

# Time Distributions
timingtotfreqs_all=pd.DataFrame(df['rt_total'])

#By Gender
timingtotfreqs_male=df[df['gender'] =="Male"]
timingtotfreqs_male=timingtotfreqs_male["rt_total"]
timingtotfreqs_female=df[df['gender'] =="Female"]
timingtotfreqs_female=timingtotfreqs_female["rt_total"]

#By Ability
timingtotfreqs_lowabil=df[df['ability'] == "Low"]
timingtotfreqs_midabil=df[df['ability'] == "Mid"]
timingtotfreqs_highabil=df[df['ability'] == "High"]
timingtotfreqs_lowabil=timingtotfreqs_lowabil["rt_total"]
timingtotfreqs_midabil=timingtotfreqs_midabil["rt_total"]
timingtotfreqs_highabil=timingtotfreqs_highabil["rt_total"]

#By Home Computer Use
timingtotfreqs_HomeY=df[df['home_computer'] == "Yes"]
timingtotfreqs_HomeN=df[df['home_computer'] == "No"]
timingtotfreqs_HomeY=timingtotfreqs_HomeY["rt_total"]
timingtotfreqs_HomeN=timingtotfreqs_HomeN["rt_total"]

# SUM SCORE Distributions
scoretotfreqs_all=pd.DataFrame(df['sum_score'])

#By Gender
scoretotfreqs_male=df[df['gender'] =="Male"]
scoretotfreqs_male=scoretotfreqs_male["sum_score"]
scoretotfreqs_female=df[df['gender'] =="Female"]
scoretotfreqs_female=scoretotfreqs_female["sum_score"]

#By HomeComputer
scoretotfreqs_HomeY=df[df['home_computer'] =="Yes"]
scoretotfreqs_HomeY=scoretotfreqs_HomeY["sum_score"]
scoretotfreqs_HomeN=df[df['home_computer'] =="No"]
scoretotfreqs_HomeN=scoretotfreqs_HomeN["sum_score"]

#Overall Timing
#4
#fig = px.histogram(df, x="rt_total", nbins=50)

#fig.update_xaxes(range=[0,5000])
#st.plotly_chart(fig)



#4A Try and do a filter bar
# -- sidebar to filter data --
st.markdown('## Overall Timing Distribution')
score_low, score_high = st.slider('Please select Sum Score range:',1, 20,(1,20))

st.markdown('Almost all respondents completed the assessment in under 1,000 seconds.  We can use the slider bar to compare high ability students who answered 19 or 20 questions correctly to those that answered fewer questions correctly.')
st.markdown('The distribution across performance levels seems roughly similar.')
df_abil = df.query('sum_score>= @score_low and sum_score <= @score_high')
#df_abil=df.query('ability_group>=@score_low and ability_group <= @score_high')
fig4 = px.histogram(df_abil,x='rt_total',nbins=50, color_discrete_sequence = ['darkred'], labels={'x':'Timing In Seconds'})
fig4.update_xaxes(range=[0,5000])
fig4.update_xaxes(title_text='Timing in Seconds')
fig4.update_yaxes(title_text='Frequency')

st.plotly_chart(fig4,use_container_width=True)

#This prints a nice overlap of 2 distributions
#5
st.markdown('## Timing by Gender: Overlay')
st.markdown('The timing distribution across Male and Female students is very similar.')

fig5 = go.Figure()
fig5.add_trace(go.Histogram(x=timingtotfreqs_male,nbinsx=50,name='Total Time: Males'))
fig5.add_trace(go.Histogram(x=timingtotfreqs_female,nbinsx=50,name='Total Time: Females'))
fig5.update_xaxes(range=[0,5000])
fig5.update_xaxes(title_text='Timing in Seconds')
fig5.update_yaxes(title_text='Frequency')

# Overlay both histograms
fig5.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig5.update_traces(opacity=0.50)
st.plotly_chart(fig5,use_container_width=True)


#6 -- sidebar to filter data --
st.markdown('## Overall Score Distribution')
age_low, age_high = st.slider('Please select age range:',18, 70,(18,70))
st.markdown('The total score distributions look similar for Males and Females, although a greater number of Male students answered all 20 questions correctly.')
st.markdown('This appears to be true for different age groups as well, though most notable for younger respondents.')



df_age = df.query('age>= @age_low and age <= @age_high')

fig6_hist = px.histogram(df_age,x='sum_score',facet_row = 'gender',color = 'gender', color_discrete_sequence=['blue','darkred'])
fig6_hist.update_xaxes(title_text='Sum Score')
fig6_hist.update_yaxes(title_text='Frequency')

st.plotly_chart(fig6_hist,use_container_width=True)

#7 NOT PRESENTED IN STREAMLIT
#st.markdown('## Score Distribution by Gender: Overlay')
#fig = go.Figure()
#fig.add_trace(go.Histogram(x=scoretotfreqs_female,name='Total Score: Females'))
#fig.add_trace(go.Histogram(x=scoretotfreqs_male,name='Total Score: Males'))
#fig.update_xaxes(range=[0,20])
#fig.update_xaxes(title_text='Sum Score')
#fig.update_yaxes(title_text='Frequency')

# Overlay both histograms
#fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
#fig.update_traces(opacity=0.50)
#st.plotly_chart(fig)

 
    
#7
st.markdown('## Scatterplot of Sum Score versus Timing')
st.markdown('The data was split into three roughly equal Ability Groups:')
st.markdown('< 18:     Low Ability')
st.markdown('18 or 19: Mid Ability')
st.markdown('20      : High Ability')
st.markdown('Almost all respondents answering all questions correctly completed the assessment in under 1,000 seconds.  Some of the low performers took a very long time or a very short time.')
TimePlot1 = px.scatter(df, x="rt_total", y="sum_score", size="sum_score", color="ability", category_orders={"ability": ["Low", "Mid", "High"]},hover_name="ability", log_x=False, size_max=15)
TimePlot1.update_xaxes(title_text='Timing in Seconds')
TimePlot1.update_yaxes(title_text='Sum Score')
st.plotly_chart(TimePlot1,use_container_width=True)

#8
st.markdown('## Scatterplot of Age Group versus Timing')
st.markdown('The size of the bubble indicates performance (larger bubble=high performance).')
st.markdown('Performance appears similar across Age Groups.  Some of the timing outliers were in the younger age groups.')
TimePlot2 = px.scatter(df, x="rt_total", y="age_group", size="sum_score", color="age_group", category_orders={"age_group": ["Age Group1: < 28", "Age Group2: 28-31", "Age Group3: 32-36","Age Group4: 37-44","Age Group5: > 45"]},hover_name="ability", log_x=False, size_max=25)
TimePlot2.update_xaxes(title_text='Timing in Seconds')
TimePlot2.update_yaxes(title_text='Age Group')
TimePlot2.update_layout(showlegend=False) 
st.plotly_chart(TimePlot2,use_container_width=True)


#9
st.markdown('## Scatterplot of Gender versus Timing')
st.markdown('The size of the bubble indicates performance (larger bubble=high performance).')
st.markdown('Performance appears similar across Gender Groups.  Most of the timing outliers were Female respondents.')
TimePlot3 = px.scatter(df, x="rt_total", y="gender_coded", size="sum_score", color="gender", color_discrete_sequence=['blue','darkred'],hover_name="gender", log_x=False, size_max=75)
TimePlot3.update_xaxes(title_text='Timing in Seconds')
TimePlot3.update_yaxes(visible=False,title_text='')
st.plotly_chart(TimePlot3,use_container_width=True)

#Means for all Timing and Scored Item variables
#df.groupby('group_column')['sum_column'].sum() 
#this gives us the # correct for gs_1

allmeans=pd.DataFrame(df.mean())
allmeansT=pd.DataFrame(allmeans.T)
allgendermeans=pd.DataFrame(df.groupby('gender').mean())
allagemeans=pd.DataFrame(df.groupby('age_group').mean())
allabilmeans=pd.DataFrame(df.groupby('ability').mean())
allhomecompmeans=pd.DataFrame(df.groupby('home_computer').mean())

print(allmeansT)
print(allgendermeans)
print(allagemeans)
print(allabilmeans)
print(allhomecompmeans)

#This is just timing and scores overall
#allmeansT_time=allmeansT.iloc[:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
allmeansT_time=allmeansT.loc[:,allmeansT.columns.str.startswith('rt_gs')]
allmeansT_scores=allmeansT.loc[:,allmeansT.columns.str.startswith('gs')]
#allmeansT_time
#allmeansT_scores

#This is all items, timing and scoring
female=allgendermeans.head(1)
male=allgendermeans.tail(1)
allmeansT=allmeansT.head(1)
allmeansT_time=allmeansT_time.head(1)
allmeansT_scores=allmeansT_scores.head(1)
female=female.to_numpy()
male=male.to_numpy()
allmeansT=allmeansT.to_numpy()
allmeansT_time=allmeansT_time.to_numpy()
allmeansT_scores=allmeansT_scores.to_numpy()
female=female.flatten()
male=male.flatten()
allmeansT=allmeansT.flatten()
allmeansT_time=allmeansT_time.flatten()
allmeansT_scores=allmeansT_scores.flatten()
print(female)
print(male)
print(allmeansT)
print(allmeansT_time)
print(allmeansT_scores)


#Try to do a histogram of all timing values overall
#10
st.markdown('## Mean Timing Across All Items')
st.markdown('Item 7, Item 9, and Item 13 took more time to answer than other items.  ')
st.markdown('Most took less than 20 seconds to answer.')
x_axis_label=  ['I2','I3','I4','I5','I6','I7','I8','I9','I10',
                'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20']
x_axis = np.arange(len(x_axis_label))
# Multi bar Chart
print(x_axis)
print(allmeansT_time)

plt.bar(x_axis, allmeansT_time, width=0.9, color ='maroon', label = 'Item Timing: All Records')
#plt.bar(x_axis +0.2, male, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()

plt.xlabel("Item")
plt.ylabel("Time in Seconds")
plt.legend().set_visible(False)
st.pyplot(plt)
plt.clf()
#Items 7, 9 and 13 had highest time

#Try to do a histogram of all item performance
#11

st.markdown('## Item Performance Across All Items')
st.markdown('Item 7, Item 9, and Item 13 took more time to answer than other items.  Item 13 was the most difficult item, while Items 7 and 9 were similarly easy.')  
st.markdown('The assessment was fairly easy with most respondents answering 80%-90% of the items correctly.')

x_axis_label=  ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10',
                'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20']
x_axis = np.arange(len(x_axis_label))
# Multi bar Chart
print(x_axis)
print(allmeansT_scores)

plt.bar(x_axis, allmeansT_scores, width=0.8, label = 'Item Difficulty: All Records')
#plt.bar(x_axis +0.2, male, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend
plt.xlabel("Item")
plt.ylabel("Proportion Correct")
plt.legend().set_visible(False)
# Display

st.pyplot(plt)
plt.clf()
#Items 7, 9 and 13 had highest time

#Examine Items 7, 9, and 13 in more detail  
#Break down timing by gender and ability level
#Break down performance by gender and age
#allgendermeans
#allabilmeans
#allagemeans

gender_T_I7_I9_I13=allgendermeans[['rt_gs_7','rt_gs_9','rt_gs_13']]
abil_T_I7_I9_I13=allabilmeans[['rt_gs_7','rt_gs_9','rt_gs_13']]

gender_S_I7_I9_I13=allgendermeans[['gs_7','gs_9','gs_13']]
abil_S_I7_I9_I13=allabilmeans[['gs_7','gs_9','gs_13']]
age_S_I7_I9_I13=allagemeans[['gs_7','gs_9','gs_13']]

#next convert so I can display side by side histograms.  It needs to be a flat 1-row array

female_T_I7_I9_I13=gender_T_I7_I9_I13.head(1)
female_T_I7_I9_I13=female_T_I7_I9_I13.to_numpy()
female_T_I7_I9_I13=female_T_I7_I9_I13.flatten()
male_T_I7_I9_I13=gender_T_I7_I9_I13.tail(1)
male_T_I7_I9_I13=male_T_I7_I9_I13.to_numpy()
male_T_I7_I9_I13=male_T_I7_I9_I13.flatten()

lowabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[0]]
lowabil_T_I7_I9_I13=lowabil_T_I7_I9_I13.to_numpy()
lowabil_T_I7_I9_I13=lowabil_T_I7_I9_I13.flatten()
midabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[1]]
midabil_T_I7_I9_I13=midabil_T_I7_I9_I13.to_numpy()
midabil_T_I7_I9_I13=midabil_T_I7_I9_I13.flatten()
higabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[2]]
higabil_T_I7_I9_I13=higabil_T_I7_I9_I13.to_numpy()
higabil_T_I7_I9_I13=higabil_T_I7_I9_I13.flatten()

#SCORE PERFORMANCE
female_S_I7_I9_I13=gender_S_I7_I9_I13.head(1)
female_S_I7_I9_I13=female_S_I7_I9_I13.to_numpy()
female_S_I7_I9_I13=female_S_I7_I9_I13.flatten()
male_S_I7_I9_I13=gender_S_I7_I9_I13.tail(1)
male_S_I7_I9_I13=male_S_I7_I9_I13.to_numpy()
male_S_I7_I9_I13=male_S_I7_I9_I13.flatten()

lowabil_S_I7_I9_I13=abil_S_I7_I9_I13.iloc[[0]]
lowabil_S_I7_I9_I13=lowabil_S_I7_I9_I13.to_numpy()
lowabil_S_I7_I9_I13=lowabil_S_I7_I9_I13.flatten()
midabil_S_I7_I9_I13=abil_S_I7_I9_I13.iloc[[1]]
midabil_S_I7_I9_I13=midabil_S_I7_I9_I13.to_numpy()
midabil_S_I7_I9_I13=midabil_S_I7_I9_I13.flatten()
higabil_S_I7_I9_I13=abil_S_I7_I9_I13.iloc[[2]]
higabil_S_I7_I9_I13=higabil_S_I7_I9_I13.to_numpy()
higabil_S_I7_I9_I13=higabil_S_I7_I9_I13.flatten()

age_1_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[0]]
age_1_S_I7_I9_I13=age_1_S_I7_I9_I13.to_numpy()
age_1_S_I7_I9_I13=age_1_S_I7_I9_I13.flatten()
age_2_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[1]]
age_2_S_I7_I9_I13=age_2_S_I7_I9_I13.to_numpy()
age_2_S_I7_I9_I13=age_2_S_I7_I9_I13.flatten()
age_3_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[2]]
age_3_S_I7_I9_I13=age_3_S_I7_I9_I13.to_numpy()
age_3_S_I7_I9_I13=age_3_S_I7_I9_I13.flatten()
age_4_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[3]]
age_4_S_I7_I9_I13=age_4_S_I7_I9_I13.to_numpy()
age_4_S_I7_I9_I13=age_4_S_I7_I9_I13.flatten()
age_5_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[4]]
age_5_S_I7_I9_I13=age_5_S_I7_I9_I13.to_numpy()
age_5_S_I7_I9_I13=age_5_S_I7_I9_I13.flatten()

#Try to do a histogram of all timing values overall
#12

st.markdown('## Looking at Items 7, 9, and 13 More Closely')
st.markdown('## Timing Distributions - By Gender')
st.markdown('Item 9 took longer for Female respondents, but the timing distributions were similar for Items 7 and 13.')  

x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
# Multi bar Chart

plt.bar(x_axis -0.2, female_T_I7_I9_I13, width=0.4, label = 'Female',  color ='maroon')
plt.bar(x_axis +0.2, male_T_I7_I9_I13, width=0.4, label = 'Male', color='blue')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend
plt.ylabel("Time in Seconds")
plt.legend()

# Display

st.pyplot(plt)
plt.clf()
#Try to do a histogram of all score values overall
#13
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
st.markdown('## Score Performance - By Gender')
st.markdown('Item 13 was easier for Male Students, while the performance was similar for Male and Female students for Items 7 and 9.')  

plt.bar(x_axis -0.2, female_S_I7_I9_I13, width=0.4, label = 'Female', color='pink')
plt.bar(x_axis +0.2, male_S_I7_I9_I13, width=0.4, label = 'Male', color='lightblue')

# Xticks

plt.xticks(x_axis, x_axis_label)
plt.ylabel("Proportion Correct")

# Add legend

plt.legend()

# Display

st.pyplot(plt)
plt.clf()
#Try to do a histogram of all timing values overall
#14
st.markdown('## Score Performance - By Ability Group')
st.markdown('Low ability students rarely answered Item 13 correctly.  With the exception of Item 13, there are many easy items that offer little discrimination amongst student abilities.')  

x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
# Multi bar Chart

plt.bar(x_axis -0.3, lowabil_S_I7_I9_I13, width=0.3, label = 'Low', color='maroon')
plt.bar(x_axis +0.0, midabil_S_I7_I9_I13, width=0.3, label = 'Mid', color='yellow')
plt.bar(x_axis +0.3, higabil_S_I7_I9_I13, width=0.3, label = 'High',color='green')


# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend(loc='lower center')

# Display
plt.ylabel("Proportion Correct")
st.pyplot(plt)
plt.clf()
#Try to do a histogram of all timing values overall
#15
st.markdown('## Score Performance - By Age Group')
st.markdown('Older students scored higher on Items 7 and 13.  Performance was similar across Age Groups for Item 9.')  
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart

plt.bar(x_axis -0.3, age_1_S_I7_I9_I13, width=0.15, label = 'Younger than 28 years')
plt.bar(x_axis -0.15, age_2_S_I7_I9_I13, width=0.15, label = '28 to 31 years')
plt.bar(x_axis +0.0, age_3_S_I7_I9_I13, width=0.15, label = '32 to 36 years')
plt.bar(x_axis +0.15, age_4_S_I7_I9_I13, width=0.15, label = '37 to 44 years')
plt.bar(x_axis +0.3, age_5_S_I7_I9_I13, width=0.15, label = 'Older than 44 years')


# Xticks

plt.xticks(x_axis, x_axis_label)
plt.ylim([0,1])
plt.ylabel('Percent Correct')
# Add legend

plt.legend(loc='lower center')

# Display
plt.ylabel("Proportion Correct")
st.pyplot(plt)
plt.clf()

# LOGISTIC REGRESSION
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#Better site
#https://realpython.com/logistic-regression-python/

#list the possible predictors, see if I can predict Item 13
# sum_score, ability_coded, home_computer_coded, age_level_coded, gender_coded are possible predictors
# dependent variable is gs_13

# import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
from sklearn import metrics

# create dummy-coded 0/1 versions of ability, age, gender, and home_computer.  

dflog=df
dflog.loc[df['ability']=='Low','ability_coded'] = 0
dflog.loc[df['ability']=='Mid','ability_coded'] = 1
dflog.loc[df['ability']=='High','ability_coded'] = 2

dflog.loc[df['age_group']=='Age Group1: < 28','age_group_coded'] = 0
dflog.loc[df['age_group']=='Age Group2: 28-31','age_group_coded'] = 1
dflog.loc[df['age_group']=='Age Group3: 32-36','age_group_coded'] = 2
dflog.loc[df['age_group']=='Age Group4: 37-44','age_group_coded'] = 3
dflog.loc[df['age_group']=='Age Group5: > 45','age_group_coded'] = 4

dflog.loc[df['gender'] == 'Female', 'gender_coded'] = 0
dflog.loc[df['gender'] == 'Male', 'gender_coded'] = 1
dflog.loc[df['home_computer'] == 'No', 'home_computer_coded'] = 0
dflog.loc[df['home_computer'] == 'Yes', 'home_computer_coded'] = 1
#dflog

#Retain the final 100 records in the datafile as a test or validation set.  Use the 1069 records to train my model
dflog_train=dflog.head(1069)
dflog_test=dflog.tail(100)

#dflog_train
#dflog_test

#MODEL 1 - JUST ABILITY CODED (LOW, MID, HIGH) AS A PREDICTOR, format it so it is an array so it can be run for logistic regression
pred1=dflog_train[['ability_coded']]
depvar=dflog_train[['gs_13']]
pred1=pred1.to_numpy().flatten()
pred1=pred1.reshape(-1,1)
depvar=depvar.to_numpy().flatten()
print(pred1)
print(depvar)

#SET UP TEST DATASET, format it so it is an array so the model can be applied to it
test1=dflog_test[['ability_coded']]
depvartest1=dflog_test[['gs_13']]
test1=test1.to_numpy().flatten()
test1=test1.reshape(-1,1)
depvartest1=depvartest1.to_numpy().flatten()
print(test1)
print(depvartest1)


#create the model, us it to predict and compare predictions to actual incorrect/correct values for Item 13"
model1 = LogisticRegression(solver='liblinear', random_state=0).fit(pred1,depvar)
print(model1.predict(test1))
print(depvartest1)
#create a confusion matrix 
confusion_matrix(depvartest1, model1.predict(test1))

#create a visualization for the Confusion Matrix

st.markdown("---")
st.markdown("## BONUS: Logistic Regression")
st.markdown("---")
st.markdown('Can the data in the dataset be used to predict performance on an item (Correct or Incorrect)?')
st.markdown('To experiment, the most difficult item (Item 13) was selected as the dependent variable and the ability group (Low, Mid, or High) was used as a predictor using Logistic Regression.  If Item 13 is a quality item with good discrimination, higher ability would predict performance on the item.')
st.markdown('100 records were withheld from the modeling set to serve as a validation set to test the model.')  

cm = confusion_matrix(depvartest1, model1.predict(test1))
st.markdown("## Confusion Matrix")
st.markdown('75% of the records were correctly classified.  The false correct / false incorrect off-diagonal misclassification counts were fairly balanced.')
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Incorrect (0)', 'Predicted Correct (1)'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Incorrect (0)', 'Actual Correct (1)'))
ax.set_xlabel('Predicted Outputs',fontsize=20)
ax.set_ylabel('Actual Outputs',fontsize=20)
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red', size=30)

st.pyplot(plt)


#MULTIPLE PREDICTORS, CAN I DO BETTER?

preds2=dflog_train[['sum_score','gender_coded','home_computer_coded']]
depvar=dflog_train[['gs_13']]
preds2=preds2.to_numpy()
depvar=depvar.to_numpy().flatten()
print(preds2)
print(depvar)

#SET UP TEST DATASET, format it so it is an array so the model can be applied to it
test2=dflog_test[['sum_score','gender_coded','home_computer_coded']]
depvartest2=dflog_test[['gs_13']]
test2=test2.to_numpy()
depvartest2=depvartest2.to_numpy().flatten()
print(test2)
print(depvartest2)

#create the model, us it to predict and compare predictions to actual incorrect/correct values for Item 13"
model2 = LogisticRegression(solver='liblinear', random_state=0).fit(preds2,depvar)
print(model2.predict(test2))
print(depvartest2)
#create a confusion matrix 
confusion_matrix(depvartest2, model2.predict(test2))

#create a visualization for the Confusion Matrix
st.markdown("## Multiple Predictors")
st.markdown('Sum Score, Gender, and Home Computer Use (Y/N) were used to predict performance on Item 13.')
st.markdown("## Confusion Matrix")
st.markdown('73% of the records were correctly classified, with a bias towards predicting a correct response when the student in fact answered incorrectly.  It turns out that a simpler model was more accurate!')
cm = confusion_matrix(depvartest2, model2.predict(test2))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Incorrect (0)', 'Predicted Correct (1)'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Incorrect (0)', 'Actual Correct (1)'))
ax.set_xlabel('Predicted Outputs',fontsize=20)
ax.set_ylabel('Actual Outputs',fontsize=20)
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red', size=30)

st.pyplot(plt)