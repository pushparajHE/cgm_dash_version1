import pandas as pd
import plotly.express as px
import streamlit as st
import altair as alt
import cgmquantify as cgm
import numpy as np
import plotly.graph_objects as go

names = ['Marcus','Manish']
usernames = ['Marcus','Manish']
passwords = ['Marcus@cgm','Manish@cgm']

pd.options.plotting.backend = "plotly"




st.set_page_config(layout='wide', initial_sidebar_state='expanded')



def calc_slope(x):
    try:
        slope = np.polyfit(range(len(x)), x, 1)[0]
    except:
        return None
    return slope

import numpy as np
def calc_slope(x):
    try:
        slope = np.polyfit(range(len(x)), x, 1)[0]
    except:
        return None
    return slope

def cal_culate_peak_stats(slope):

    if slope.shape[0]==1:
        return {}
    slope=slope.sort_values('Actual time')
    max_bg=slope['Glucose reading'].max()
    peak_time=slope[slope['Glucose reading']==max_bg]['Actual time'].max()
    upslope=slope[slope['Actual time']<=peak_time]
    down_slope=slope[slope['Actual time']>peak_time]
    stats={}
    stats['time_to_reach_top']=(upslope['Actual time'].max()-upslope['Actual time'].min()).total_seconds() / 60.0
    stats['time_to_come_down']=(down_slope['Actual time'].max()-down_slope['Actual time'].min()).total_seconds() / 60.0
    stats['start_time']=upslope['Actual time'].min()
    stats['end_time']=upslope['Actual time'].max()
    stats['max_BG']=max_bg
    stats['Total_sample']=slope.shape[0]
    stats['upslope_sample']=upslope.shape[0]
    stats['up_start_BG']=upslope['Glucose reading'].min()
    stats['down_end_BG']=down_slope['Glucose reading'].min()
    stats['downslope_sample']=down_slope.shape[0]
    stats['peak_time']=peak_time
    return stats


def preprocess_data(cgm_df):
    format = '%d-%m-%Y %H:%M%p'
    try:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'], format=format)
    except:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'])


    cgm_df = cgm_df[['Actual time', 'Glucose reading']]
    df = cgm_df[cgm_df['Glucose reading'].notnull()]
    df = df.sort_values('Actual time')
    df['diff_next_sample'] = df['Actual time'].diff().dt.days
    df['tag_no'] = df['diff_next_sample'] > 1
    df['tag_no'] = df['tag_no'].cumsum()
    #df['month'] = df['Actual time'].dt.to_period('M').astype(str)
    df['Tag No'] = df.tag_no.apply(lambda x: f'Tag_No - {x+1}')


    return df


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


def perform_analysis(glucose):

    glucose = glucose.sort_values('Actual time')
    glucose = glucose[['Actual time', 'Glucose reading']]
    import pandas as pd
    glucose['Actual time'] = pd.to_datetime(glucose['Actual time'])
    glucose['Day']=glucose['Actual time'].dt.date
    glucose['Time']=glucose['Actual time'].dt.hour
    glucose['weekDay'] = glucose['Actual time'].dt.day_name()
    glucose=glucose.reset_index(drop=True)
    
    date1=str(glucose["Actual time"].min().date())
    date2 = str(glucose["Actual time"].max().date()) 
    date3 = str((glucose["Actual time"].max() - glucose["Actual time"].min()).days)
    


   # Glucose Matrices calculation
    
    
    

       
    
    #time_spent _ingroup=
    



    # m1.metric(label='FROM', value='',delta=str(glucose["Actual time"].min().date()))
    GMI_HbA1c=np.round(cgm.GMI(glucose),3)
    ADA_HbA1c=np.round(cgm.eA1c(glucose),3)
    pTIR=np.round(cgm.PInsideRange(glucose), 3)
    Adrr=np.round(cgm.ADRR(glucose), 3)



    glucose['upper'] = glucose['Glucose reading'] > 110
    glucose['previuos'] = glucose['upper'].shift()
    glucose['next'] = glucose['upper'].shift(-1)
    glucose[['upper', 'previuos', 'next']] = glucose[['upper', 'previuos', 'next']].fillna(False)
    glucose['isupper'] = glucose.apply(lambda x: (x['upper'] | x['previuos'] | x['next']), axis=1)

    upper = glucose[glucose.isupper]


    upper['is_new_segment'] = upper.reset_index()['index'].diff().values > 1
    upper['peak'] = upper['is_new_segment'].cumsum()

    max_BG_peak = upper.groupby('peak')['Glucose reading'].max().reset_index()

    peak_day = upper.groupby(upper['Actual time'].dt.date)['peak'].nunique().reset_index()
    peak_day.columns = ['Date', 'Total Peaks']

    time_spent_in_peak = upper.groupby('peak')['Actual time'].apply(
        lambda x: (max(x) - min(x)).total_seconds() / 60.0) + 1
    time_spent_in_peak = time_spent_in_peak.reset_index()

    time_spent_in_peak.columns = ['Peak No', 'Time in spike (Min)']


    # spikes average calculation

    

    AvgMax_in_peak=np.round(np.mean(max_BG_peak["Glucose reading"]),2)
    Average_peaks_eachday=np.round(np.mean(peak_day["Total Peaks"]),2)
    average_time_spent_in_peaks=np.round(np.mean(time_spent_in_peak["Time in spike (Min)"]),2)

    average_bg=np.round(np.mean(glucose[ 'Glucose reading']),2)

    
    # spikes Stats calculation

    max_BG_peak.columns = ['Spike No', 'Max Glucose']
    peak_day.columns = ['Date', 'Spikes']
    time_spent_in_peak.columns = ['Spike No', 'Time in spike (Min)']

    stats = upper.groupby('peak').apply(cal_culate_peak_stats)
    import pandas as pd
    stats_df = pd.DataFrame.from_dict(list(stats.values))
    stats_df['peak'] = stats.index
    stats_df = stats_df.rename(columns={'peak': 'Spike No'})
    stats_df = stats_df[stats_df.upslope_sample != 1]
    stats_df['Rising_Rate'] = stats_df.apply(lambda x: (x['max_BG'] - x['up_start_BG']) / x['time_to_reach_top'],
                                             axis=1)

    stats_df['Falling_Rate'] = stats_df.apply(
        lambda x: (x['max_BG'] - x['down_end_BG']) / x['time_to_come_down'] if x['time_to_come_down'] != 0 else 0,
        axis=1)
    import pandas as pd
    stats_df = pd.merge(stats_df, time_spent_in_peak, on='Spike No', how='left')
    stats_df = pd.merge(stats_df, max_BG_peak, on='Spike No', how='left')
    stats_df = stats_df.rename(columns={'peak_time': 'Spike Time'})

    cols = ['Spike No', 'time_to_reach_top', 'time_to_come_down', 'Rising_Rate', 'Falling_Rate', 'Time in spike (Min)',
            'Max Glucose', 'start_time', 'end_time', 'max_BG', 'Total_sample', 'upslope_sample', 'up_start_BG',
            'down_end_BG'
        , 'downslope_sample', 'Spike Time']
    stats_df = stats_df[cols]

    # For Crashes
    glucose['lower'] = glucose['Glucose reading'] < 70
    glucose['L_previuos'] = glucose['lower'].shift(-1)
    glucose['L_next'] = glucose['lower'].shift()
    glucose[['lower', 'L_previuos', 'L_next']] = glucose[['lower', 'L_previuos', 'L_next']].fillna(False)
    glucose['islower'] = glucose.apply(lambda x: (x['lower'] | x['L_previuos'] | x['L_next']), axis=1)

    lower = glucose[glucose.islower] 

    lower['is_new_segment'] = lower.reset_index()['index'].diff().values > 1
    lower['crash'] = lower['is_new_segment'].cumsum()


    crash_day = lower.groupby(lower['Actual time'].dt.date)['crash'].nunique().reset_index()
    crash_day.columns = ['Date', 'crash']






    ##### Rader Chart
    ATtoFloorfrom_Peak = np.round(stats_df[stats_df.time_to_come_down > 0]['time_to_come_down'].mean(), 2)
    ATtoTopfrom_base = np.round(stats_df[stats_df.time_to_come_down > 0]['time_to_reach_top'].mean(), 2)

    ## Average time in spike add to DB
    
   

    Average_peaks_eachdayR = [6, 5, 4, 3]
    ATtoFloorfrom_PeakR = [151, 91, 60, 31]
    average_time_spent_in_peaksR = [67, 51, 18, 19, 3]
    ADRRR = [50, 40, 30, 19, 10]
    GMI_HbA1cR = [8, 6.5, 5.7, 5]

    AvgMax_in_peakR=[181,180,150,140]
    average_bgR=[180,140,130,110]




    def get_score(range_val, val):
        for i in range(len(range_val)):
            if val >= range_val[i]:
                return i+1
        return 5

    TINRR = [40, 60, 75, 95]
    #ATtoTopfrom_baseR = [46, 47, 48, 49, 50]


    def get_score_reverse(range_val, val):
        for i in range(len(range_val)):
            if val <= range_val[i]:
                return i+1
        return 5

    AvgMax_in_peak_Score=get_score(AvgMax_in_peakR, AvgMax_in_peak)
    Average_peaks_eachday_Score = get_score(Average_peaks_eachdayR, Average_peaks_eachday)
    ATtoFloorfrom_PeakR_score = get_score(ATtoFloorfrom_PeakR, ATtoFloorfrom_Peak)
    average_time_spent_in_peaks_score = get_score(average_time_spent_in_peaksR, average_time_spent_in_peaks)

    average_bg_Score=get_score(average_bgR,average_bg)

    ADRR_score = get_score(ADRRR, Adrr)
    TINR_score = get_score_reverse(TINRR, pTIR)
    #ATtoTopfrom_base_score = get_score_reverse(ATtoTopfrom_baseR, ATtoTopfrom_base)
    GMIA1c_Score = get_score(GMI_HbA1cR, GMI_HbA1c)

    CGM_index_Score = (0.1 * GMIA1c_Score) + (0.15 * Average_peaks_eachday_Score) + (
                0.15 * average_time_spent_in_peaks_score) + (0.1 * ATtoFloorfrom_PeakR_score) + (
                                  0.2 * AvgMax_in_peak_Score) + (0.2 * TINR_score) + (0.1 * average_bg_Score)

    CGM_index_Score= str(np.round(CGM_index_Score,3))
    avgtinspike = f'{average_time_spent_in_peaks} Min'

    #Add Start end date to dashboard
    dt1,  dt2,  dt3, dt4 = st.columns((1, 1, 1, 1))

    dt1.markdown("**:blue[FROM]**")#('FROM')
    dt1.write(date1)
    


    dt2.markdown('**:blue[TO]**')
    dt2.write(date2)


    dt3.markdown('**:blue[TOTAL DAYS]**')
    #dt3.write(str((glucose["Actual time"].max() - glucose["Actual time"].min()).days))
    dt3.write(date3)
    
    dt4.markdown('**:blue[Human Edge Score]**')
    dt4.write(CGM_index_Score)
    st.markdown("""---""")
    
    grop1 = len(glucose[glucose['Glucose reading'].between(161,250)])
    grop2 = len(glucose[glucose['Glucose reading'].between(131,160)])
    grop3 = len(glucose[glucose['Glucose reading'].between(90,130)])
    grop4 = len(glucose[glucose['Glucose reading'].between(75,89)])
    grop5 = len(glucose[glucose['Glucose reading'].between(40,69)])
    
    glucose['HbA1c'] = (46.7 + glucose['Glucose reading']) / 28.7
    glucose['day/night'] = glucose['Time'].apply(lambda x: 'night' if 0 <= x <= 7 or 20 <= x <= 23 else 'day')
    comp_glucose =glucose['Glucose reading']
    YY = []
    for k in range(len(comp_glucose)):
            if np.all(comp_glucose[k]>160):
                  YY.append('very_high')
            elif np.all(130<comp_glucose[k]<=160):
                 YY.append('high')
            elif np.all(90<comp_glucose[k]<=130):
                YY.append('Normal')
            elif np.all(75<comp_glucose[k]<=90):
                YY.append('low')
            else:
                YY.append('very_low')
    glucose['group_status'] =YY      
    #import plotly.graph_objects as go
    #st.dataframe(data=glucose, width=1024, height=768)
    unique_very_high = glucose[glucose['group_status'] == 'very_high']['Day'].nunique()
    unique_very_low = glucose[glucose['group_status'] == 'very_low']['Day'].nunique()
    unique_high = glucose[glucose['group_status'] == 'high']['Day'].nunique()
    unique_low = glucose[glucose['group_status'] == 'low']['Day'].nunique()
    
    zy=[]
    for j in range(len(comp_glucose)):
            if np.all(80<comp_glucose[j]<=130):
                  zy.append('nor')
            else :
                zy.append('hyp')
                
    glucose['nor/hyp'] =zy
    unq_all = glucose['Day'].nunique()
    unique_fke = glucose[glucose['nor/hyp'] == 'hyp']['Day'].nunique()
    unique_nor = unq_all - unique_fke
    
    zz=[]
    for l in range(len(comp_glucose)):
            if np.all(comp_glucose[l]>130):
                  zz.append('hyper')
            elif np.all(80<comp_glucose[l]<=130):
                 zz.append('normal')
            else :
                zz.append('hypo')
                
    glucose['hypr/hypo'] =zz         
    unique_hyper = str(glucose[glucose['hypr/hypo'] == 'hyper']['Day'].nunique())      
    unique_hypo = str(glucose[glucose['hypr/hypo'] == 'hypo']['Day'].nunique())
     
    
    #st.dataframe(data=glucose, width=1024, height=768)
    glucose2=glucose[['Actual time','Day','group_status','day/night']]
    import altair as alt
    #from vega_datasets import data
    
    #glucose value aslysis
    avg_glucose = round(glucose['Glucose reading'].mean(),2)
    high_glucose= round(glucose['Glucose reading'].max(),2)
    low_glucose = round(glucose['Glucose reading'].min(),2)
    night_averageG = round(glucose[glucose['day/night']=='night']['Glucose reading'].mean(), 2)
    day_averageG = round(glucose[glucose['day/night']=='day']['Glucose reading'].mean(), 2)
    
    Labels= ['Very High(161<)', 'High(131-160)', 'Normal(91-130)', 'Low(75-90)', 'Very Low(74>)']
    sizes= [grop1, grop2, grop3, grop4, grop5]
    #explode = (0.4, 0, 0, 0, 0.4)
    
    colval, colpie = st.columns([1,3])
       
    figz= px.pie(sizes, values=sizes, names = Labels, color=Labels, title = 'Pie chart for time range',
                 color_discrete_map={
                'Very High(161<)': "red",
                "Normal(91-130)": "#49B55E",
                'Very Low(74>)': "#6495ED",
                "Low(75-90)": "#191970",
                'High(131-160)': "#FF7F0E"})
    figz.update_traces(textposition= 'inside', textinfo= 'percent+label', hole=.5)
    figz.update_layout(title_font_size = 42, autosize=False, width=500, height=400, margin=dict(l=10, r=10, t=20, b=5) )
    colpie.plotly_chart(figz)
    #source = data.seattle_weather()
    colval.metric("Average glucose:", avg_glucose)
    colval.metric("Max glucose:", high_glucose)
    colval.metric("min glucose:", low_glucose)
    colval.metric("Average at night", night_averageG)
    colval.metric("Average at day", day_averageG)
    st.markdown("""---""")
    
    
    totalgrop = grop1 + grop2 + grop3 + grop4 + grop5
    Pgrop1= round(grop1/totalgrop*100,2)
    Pgrop2= round( grop2/totalgrop*100,2)
    Pgrop3= round( grop3/totalgrop*100,2)
    Pgrop4= round( grop4/totalgrop*100,2)
    Pgrop5= round( grop5/totalgrop*100,2)
    
    
    valg1 = "{}%".format(Pgrop1)
    valg2 = "{}%".format(Pgrop2)
    valg3 = "{}%".format(Pgrop3)
    valg4 = "{}%".format(Pgrop4)
    valg5 = "{}%".format(Pgrop5)
    
    with st.expander(" **analysis for group status**"):
        pg1, pg2, pg3, pg4, pg5 = st.columns(5)
         
        pg1.metric('Very High(161<)', valg1)
        pg1.metric('days(atlist)', unique_very_high )
        
        pg2.metric("High(131-160)", valg2)
        pg2.metric('days(atlist)', unique_high )
        pg3.metric("Normal(91-130)", valg3)
        pg3.metric('days(only)', unique_nor )
        pg4.metric("Low(75-90)", valg4)
        pg4.metric('days(atlist)', unique_low )
        pg5.metric("Very Low(74>)", valg5 )
        pg5.metric('days(atlist)', unique_very_low )

        st.write('Number of days spent with Hyperglycemia:', unique_hyper,  )
        st.write('Number of days spent with Hypoglycemic:', unique_hypo, )
    
        
    
    import base64
    tabh, tabz, tabwk =st.tabs(["Glucose Analysis", "Group Pattern Analysis", "Day/Night Analysis", ])
    with tabh:
         
         fig_group = px.scatter(glucose, x="Actual time", y="Glucose reading", color="group_status" , 
                     color_discrete_map={
                        "very_high": "red",
                        "high": "#FF7F0E",
                        "Normal": "#49B55E",
                        "low": "#0D2A63",
                        "very_low": "skyblue"},
                        title="CGM Group Analysis"
                        )
         st.plotly_chart(fig_group, use_container_width=True )
         # stdev for glucose reading 
         line1 = alt.Chart(glucose).mark_line().encode(
            x='monthdate(Day):O',
            y='mean(Glucose reading)'
         )
    
         band2 = alt.Chart(glucose, title='Glucose reading stdev').mark_errorband(extent='stdev').encode(
            x='monthdate(Day):O',
            y=alt.Y('Glucose reading', title='Glucose reading(mg/dL)'),
         )
    
         chart7 = band2 + line1
         st.altair_chart(chart7, theme="streamlit", use_container_width=True)    

         
    with tabz:
         #_____________chart weekly day__________________
         scale = alt.Scale(
         domain=['very_high', 'high', 'Normal', 'low', 'very_low'],
         range=["#CD5C5C", "#A52A2A", "#49B55E", "#191970", "#6495ED"],)
         color = alt.Color('group_status:N', scale=scale)
         
         # bar chart for weekly
         bar_rounded = alt.Chart(glucose).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, 
             ).encode(
                 x=alt.X('day(Day):N',  ),
                 y='count():Q',
                 color=color
             
                 ).properties(width=50)
                 
         #st.altair_chart(bar_rounded, theme="streamlit", use_container_width=True )
         
         
         fipB= px.bar(glucose,  x = 'weekDay', color = 'group_status',
                      color_discrete_map={
                     "very_high": "red",
                     "Normal": "#49B55E",
                     "very_low": "#6495ED",
                     "low": "#191970",
                     "high": "#FF7F0E"})
         fipB.update_layout(barmode='stack', yaxis={'categoryorder':'category ascending'})
         fipB.update_xaxes(categoryorder='array', categoryarray= ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
         fipB.update_yaxes(categoryorder='array',categoryarray= [ 'very_high','high','very_low','low', 'Normal'])
         fipB.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                       plot_bgcolor = "rgba(0,0,0,0)")
         st.write(fipB)

         # ------------------box plot night and day_____________
         fbox = px.box(glucose, x= 'group_status', y= 'Glucose reading' , color= 'day/night',
                       color_discrete_map={
                      "very_high": "red",
                      "Normal": "#49B55E",
                      "day": "#49B55E",
                      "night": "#191970",
                      "high": "#FF7F0E"})
         fbox.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                       plot_bgcolor = "rgba(0,0,0,0)")
         st.write(fbox)

         
         #____________________________
         scale = alt.Scale(
         domain=['very_high', 'high', 'Normal', 'low', 'very_low'],
         range=["#FF0000", "#A52A2A", "#098616", "#191970", "#6495ED"],)
         color = alt.Color('group_status:N', scale=scale)     

         #-------------------------INTERACTIVE CHART--------------
        # interactive glucose reading analysis
         base = alt.Chart(glucose).properties(width=900)
         selection = alt.selection_multi(fields=['group_status'], bind='legend')
         line = base.mark_line().encode(
             x='Actual time',
             y=alt.Y('Glucose reading', title='Glucose reading(mg/dL)'),
             color=color,
             strokeDash='group_status',
             opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
         ).add_selection(selection)

         rule = base.mark_rule().encode(
             y='average(Glucose reading)',
             color='group_status',
             size=alt.value(2)
         )

         chart = line 

         
         selection1 = alt.selection_multi(fields=['group_status'])
         chart2 = alt.Chart(glucose).mark_bar().encode(
            x='count()',
            y='group_status:N',
            color=color,
          ).add_selection(selection1)
          

          
         chart4 = alt.vconcat(chart, chart2, data = glucose, title = "glucose reading analysis" )
         st.altair_chart(chart4, theme= None, use_container_width= True)
         
         # table interactive
         gener_group = st.radio("Please select group",('Very High', 'High', 'Normal', 'Low', 'Very Low'), horizontal=True) 
         
         if gener_group == 'Very High':
             glucose_group = glucose2.loc[glucose2['group_status']=='very_high']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_Very_high.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)

         elif gener_group == 'High':
             glucose_group = glucose2.loc[glucose2['group_status']=='high']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_High.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)            
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='high'], width=1024, height=768)
             
         elif gener_group == 'Normal':
             glucose_group = glucose2.loc[glucose2['group_status']=='Normal']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_Normal.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)        
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='Normal'], width=1024, height=768)
             
         elif gener_group == 'Low':
             glucose_group = glucose2.loc[glucose2['group_status']=='low']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_Low.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768) 
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='low'], width=1024, height=768) 
             
         else :
             glucose_group = glucose2.loc[glucose2['group_status']=='very_low']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_High.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768) 
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='very_low'], width=1024, height=768)

                 
    with tabwk:
        
          
         fbox = px.violin(glucose, x= 'day/night', y= 'Glucose reading' , color= 'day/night', box=True,
                       color_discrete_map={
                      "very_high": "red",
                      "Normal": "#49B55E",
                      "day": "#49B55E",
                      "night": "#191970",
                      "high": "#FF7F0E"})
         fbox.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                       plot_bgcolor = "rgba(0,0,0,0)")
         st.write(fbox) 
         
         #candlestick chart 
         chartC = alt.Chart(data= glucose, title = "glucose reading analysis_ candlestick").mark_boxplot(extent='min-max').encode(
          x='hours(Actual time):O',
           y=alt.Y('Glucose reading', title='Glucose reading(mg/dL)')
             )

         st.altair_chart(chartC, theme="streamlit", use_container_width=True)

         

         
    
    
    
    #candlestick chart for glucose
         scale1 = alt.Scale(
         domain=['day','night'],
         range=[ "#49B55E", "#191970"],)
         color1 = alt.Color('day/night:N', scale=scale1)
    
         bar_rounded1 = alt.Chart(glucose, title='Glucose reading day/night candlestick ').mark_boxplot(extent='min-max'
             ).encode(
                x='monthdate(Day):O',
                y='average(Glucose reading)',
                color=color1,
                ).properties(width=50)
                
         st.altair_chart(bar_rounded1, theme=None, use_container_width=True )  
        
        
         bar_rounded2 = alt.Chart(glucose,title='Glucose reading day/night line chart').mark_line(
            ).encode(
                x='monthdate(Day):O',
                y='average(Glucose reading)',
                color=color1,
                ).properties(width=50)
              
         st.altair_chart(bar_rounded2 , theme=None, use_container_width=True ) 
        
        
    
    
    
    
    
        
 
             
    
    
         
         
    
    
    #tabs header
    
    tab1, tab2, tab3 = st.tabs(["HbA1", "Spike", "HE Score"])
    with tab1:
         st.subheader("Glucose Matrices calculation")
         gm2, gm3, gm4  = st.columns((1, 1, 1))
         gm2.metric("American Diabetes Association-HbA1c", ADA_HbA1c )
         gm3.metric("Average Daily Risk Range", Adrr )
         gm4.metric("Percent time inside range 70-150", pTIR )
         
         #gm1.markdown('**Glucose management index-HbA1c**')
         #gm1.subheader(GMI_HbA1c)
         #gm2.markdown('**American Diabetes Association-HbA1c**')
         #gm2.subheader(ADA_HbA1c)
         #gm3.markdown('**Average Daily Risk Range**')
         #gm3.subheader(Adrr)
         #gm4.markdown('**Percent time inside range 70-150**')
         #gm4.subheader(pTIR)
         with st.expander("**Estimated Daily HbA1c**"):
             fig_hb = px.bar(glucose, x='Day', y= 'HbA1c')
             fig_hb.update_traces(marker_color='#191970')
             fig_hb.update_layout(title_text="Estimated Daily HbA1c", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
             #st.plotly_chart(fig_hb)
             chart_Hb = alt.Chart(glucose).mark_area(
                color="lightblue",
                interpolate='step-after',
                    line=True
                    ).encode(
                    x='monthdate(Day):O',
                    y='average(HbA1c)'
                    )
             st.altair_chart(chart_Hb, theme="streamlit", use_container_width=True)
             st.write(''' HbA1c, also known as glycated hemoglobin, is a measure of the average blood glucose levels over the past 2-3 months. It is expressed as a percentage and is used to monitor the 
                      effectiveness of diabetes management over time. The higher the HbA1c level, the greater the risk of developing diabetes-related complications.''')
                      
                     
    with tab2:
        st.header("spikes average calculation")
        pm1, pm2, pm3 = st.columns((1, 1, 1))

        pm1.markdown('**Daily Average spike (>110)**')
        pm1.subheader(Average_peaks_eachday)

        pm2.markdown('**Average Max BG in spike**')
        pm2.subheader(AvgMax_in_peak)
        pm3.markdown('**Average Time in spike**')
        pm3.subheader(avgtinspike)

        st.markdown("""---""")
        with st.expander(" **Time Spent in Spike**"):
            time_spent_in_peak = pd.merge(time_spent_in_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')
            time_spent_in_peak = time_spent_in_peak[time_spent_in_peak['Time in spike (Min)'] < 600]
            fig3 = px.line(time_spent_in_peak, x="Spike No", y='Time in spike (Min)', hover_data={'Spike Time': True})
            fig3.update_traces(marker_color='#d40707')
            fig3.update_layout(title_text="Time Spent in Spike", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig3)

        st.markdown("""---""")

        with st.expander(" **Spikes in a Day**"):
            fig1 = px.bar(peak_day, x="Date", y='Spikes')
            fig1.update_traces(marker_color='#191970')
            fig1.update_layout(title_text="Spikes in a Day", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig1)
            st.write('''
            The chart above shows some numbers.
            It rolled actual dice for these, so they're guaranteed to  be random.''')
            
        with st.expander("**Crashes in Day**"):
            fig_crash = px.bar(crash_day, x='Date', y= 'crash')
            fig_crash.update_traces(marker_color='#191970')
            fig_crash.update_layout(title_text="Crashes in a Day", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig_crash)
            st.write(''' crashes of glucose reading less than 70 mg/dL.''')      
            
            
    with tab3:
        st.header("hes")
        ATTF, ATTTOP,AverageBG_DB = st.columns((1, 1, 1))

        ATTF.markdown('**Average time to Peak (Min)**')
        ATTF.subheader(ATtoFloorfrom_Peak)

        ATTTOP.markdown('**Average time to floor (Min)**')
        ATTTOP.subheader(ATtoTopfrom_base)

        AverageBG_DB.markdown('**Average BG**')
        AverageBG_DB.subheader(average_bg)
    
    
    
    
    
    
    
    

    import pandas as pd
    df = pd.DataFrame(dict(
        score=[Average_peaks_eachday_Score, ADRR_score, TINR_score, average_time_spent_in_peaks_score
             , ATtoFloorfrom_PeakR_score, AvgMax_in_peak_Score,average_bg_Score],
        metric=['Daily Average spike (>110)', 'Average Daily Risk Range',
                'Percent time inside range 70-150', 'Average Time in spike', 'Average time to Peak',
                'Average Maximum in each peak','Average Blood Glucose'
                ]))





    #Redarfig = px.scatter(df, y='score', x='metric',  size_max=60)
    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    st.markdown("""---""")

    st.subheader("Metrices Score :")
    #st.plotly_chart(Redarfig, use_container_width=True,theme="streamlit")
    st.plotly_chart(Redarfig, use_container_width=True)
    # Line chart ploting on dash board

    st.markdown("""---""")

    genred = st.sidebar.selectbox("Select Patient group",('Normal person without diabetes','Official ADA recommendation for a person with diabetes') )
    
    if genred=='Official ADA recommendation for a person with diabetes':
       glucose['HE 130 mg/dL']= 130
       glucose['HE 80 mg/dL'] = 80
       fig_GR = px.line(glucose, x="Actual time", y=["Glucose reading", 'HE 130 mg/dL', 'HE 80 mg/dL'])
       
       fig_GR.update_traces(marker_color='#191970')
       #fig_GR.add_trace(go.Scatter(x="Actual time", y=["Glucose reading", 'HE 150', 'HE 70'],line_color='rgb(0,100,80)'))
       fig_GR.update_layout(title_text="Glucose Reading", title_x=0, margin=dict(l=0, r=15, b=15, t=50),
                            yaxis_title='Glucose mg/dL', xaxis_title='datetime')
       fig_GR.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                           plot_bgcolor = "rgba(0,0,0,0)")
       st.plotly_chart(fig_GR, use_container_width=True)
 
       st.write(''' The American Diabetes Association (ADA) recommends that people with diabetes check their blood glucose levels at 
                regular intervals to help manage their condition. The recommended frequency and target ranges for glucose levels vary depending on the type of diabetes, treatment plan, and individual needs.''')   
 
       # bio hacks
       with st.expander(" **Biohack for Diabetes**"):
           st.text(''' If you get sick, your blood sugar can be hard to manage.''')
           
           #local_df = pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
           values=[['Setup up your meals: fibre, protein, and healthy fats', 'Choose foods with low glycemic index', 'Zone 2 after meals', 
                    "Eat more lower GI foods", 'Eat foods rich in chromium'], 
                   ["Plan your meals to include fibre, protein and healthy fats to help decrease blood sugar spikes after a meal", 
                    "Choose to eat foods with a low glycemic index (GI) to monitor your overall carbohydrate intake and keep your blood glucose levels more stable throughout the day.", 
                    "Go for a 20-30 minute walk around the block after meals",
                    "Swap out high GI foods with low GI foods - for example white bread for whole wheat or rye bread, or white rice for brown rice.",
                    "Take 200-1,000 mcg of chromium picolinate to boost micronutrient levels, stabilise blood glucose levels, and aid against diabetes."]]
           
           figT= go.Figure(data= [go.Table( columnwidth=(250),

               header = dict(
                   values = [['<b>Biohack title</b>'],
                     ['<b>Biohack</b>']],
                   line_color='darkslategray',
                   fill_color='royalblue',
                   align=['center','center'],
                   font=dict(color='white', size=12),
                   height=40),
                  cells=dict(
                    values=values,
                    line_color='darkslategray',
                    fill=dict(color=['paleturquoise', 'white']),
                    align=['center', 'left'],
                    font_size=10,
                    height=40)
                    )           
                 ])
           st.write(figT)
       
    else :
       glucose['HE 99 mg/dL']= 99
       glucose['HE 70 mg/dL'] = 70
       fig_GR = px.line(glucose, x="Actual time", y=["Glucose reading", 'HE 99 mg/dL', 'HE 70 mg/dL'])
       
       fig_GR.update_traces(marker_color='#191970')
       #fig_GR.add_trace(go.Scatter(x="Actual time", y=["Glucose reading", 'HE 150', 'HE 70'],line_color='rgb(0,100,80)'))
       fig_GR.update_layout(title_text="Glucose Reading", title_x=0, margin=dict(l=0, r=15, b=15, t=50),
                            yaxis_title='Glucose mg/dL', xaxis_title='datetime')
       fig_GR.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                           plot_bgcolor = "rgba(0,0,0,0)")
       st.plotly_chart(fig_GR, use_container_width=True)
 
       st.write(''' A normal glucose reading is generally considered to be between 70 and 99 mg/dL when measured fasting, or before a meal. Blood glucose levels tend to be higher after meals and can reach up to 120 mg/dL or slightly higher. ''')


       # bio hacks
       with st.expander(" **Biohack for Diabetes**"):
           st.text(''' If you get sick, your blood sugar can be hard to manage.''')
           st.image("CGM-Be.png", width=330)
           #local_df = pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
           values=[['Replace artificial sweetners with coconut sugar', 'Swap your dessert with a whole fresh fruit', 'Eat vegetables first', 'Teaspoon of cinnamon pre meals reduces 1st phase insulin response', 
                    'Teaspoon of apple cider vinegar pre bedtime reduces 1st phase insulin response'], 
                   ["Manage your glucose spikes by replacing artificial sweeteners with coconut sugar in your cup of tea or coffee.",
                    "Manage your glucose spikes by swapping your dessert with whole fresh fruit (with skin) to increase your fibre intake and manage your calories.", 
                    "Eat your vegetables first before you eat your meat. Eat carbs last.", 
                    "Manage your post-prandial blood glucose spike by having a teaspoon of cinnamon in a  glass of water before meals.", 
                    "Manage your bedtime blood glucose levels by having a teaspoon of apple cider vinegar in a glass of water before bedtime."]]
           
           figT= go.Figure(data= [go.Table( columnwidth=(250),

               header = dict(
                   values = [['<b>Biohack title</b>'],
                     ['<b>Biohack</b>']],
                   line_color='darkslategray',
                   fill_color='royalblue',
                   align=['center','center'],
                   font=dict(color='white', size=12),
                   height=40),
                  cells=dict(
                    values=values,
                    line_color='darkslategray',
                    fill=dict(color=['paleturquoise', 'white']),
                    align=['center', 'left'],
                    font_size=10,
                    height=40)
                    )           
                 ])
           st.write(figT)  
    
    
        
        
    with st.expander(" **Glucose Reading explanation**"):
        st.text(''' If you get sick, your blood sugar can be hard to manage. You may not be able to eat or drink as much as usual, 
which can affect blood sugar levels. If you're ill and your blood sugar is 240 mg/dL or above, use an over-the-counter ketone test kit to 
check your urine for ketones and call your doctor if your ketones are high. High ketones can be an early sign of diabetic ketoacidosis, 
which is a medical emergency and needs to be treated immediately''')
        
        
    #st.info(f'Peaks Max BG Average : **{}**')
    st.markdown("""---""")
    with st.expander(" **Maximum Glucose in spike**"):
        max_BG_peak = pd.merge(max_BG_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')
    
        fig2 = px.line(max_BG_peak, x="Spike No", y="Max Glucose",hover_data={'Spike Time':True})
        fig2.update_traces(marker_color='#d40707')
        fig2.update_layout(title_text="Maximum Glucose in spike", title_x=0, margin=dict(l=0, r=15, b=15, t=50))
        st.plotly_chart(fig2)

    

    

    #fig=upper.groupby(upper['Actual time'].dt.date)['peak'].nunique().plot()
    st.markdown("""---""")
    #st.markdown("Glucose_readings throughout a day")
    #st.image("gcm_day.png", width=800)
    #fig = px.line(glucose,x="Actual time", y="Glucose reading", title="Blood Glucose Timeline")
    #st.plotly_chart(fig)
    

    


    st.download_button(
    "Press to Download table",
    convert_df(stats_df),
    "cgm_stats.csv",
    "text/csv",
    key='download-csv')
    st.dataframe(data=stats_df, width=1424, height=450)
    
    
    #PDFbyte = doc1.read()
    
     
    
    
      

def open_dash(df):
    cgmdf = preprocess_data(df)

    cgmdf['date'] = pd.to_datetime(cgmdf['Actual time'].dt.date)
    tags_dates = cgmdf.groupby('Tag No')['date'].agg({'min', max}).reset_index()

    tags_dates.columns = ['Tag No', 'start date', 'end date']
    #st.sidebar.dataframe(data=tags_dates)
    tags = list(cgmdf['Tag No'].unique())

    tagnum = st.sidebar.selectbox("Select tag Number:", tags)

    glucose = cgmdf[cgmdf['Tag No'] == tagnum]

    daterange = st.sidebar.date_input("Select Tag Date Range",
                                      (glucose['date'].min(), glucose['date'].max()), min_value=glucose['date'].min(),
                                      max_value=glucose['date'].max())
    # st.write(type(daterange[0]))
    # st.write(type(glucose['date'].dtypes))
    if len(daterange) == 2:
        glucose = glucose[
            (glucose['date'] >= pd.to_datetime(daterange[0])) & (glucose['date'] <= pd.to_datetime(daterange[1]))]
        perform_analysis(glucose)





def app():

    with st.sidebar.header('Upload your CGM data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        unit_button = st.sidebar.selectbox("Select units:", ('mg/dL','mmol/L'))

    @st.cache(allow_output_mutation=True)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    st.sidebar.title("**Metabolic**")


    # Web App Title
    st.title(':purple[Continuous Glucose Monitoring]')

    if uploaded_file is not None:
        df = load_csv()  # pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(df)
    else:
        local_df = pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(local_df)






