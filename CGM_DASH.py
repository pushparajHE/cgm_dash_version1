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

    cgm_df = cgm_df[['Actual time', 'Glucose reading', 'Event type / Notif type', 'Event name']]
    cgm_df['Event type / Notif type'] = cgm_df['Event type / Notif type'].shift(-1)
    cgm_df['Event name'] = cgm_df['Event name'].shift(-1)
    df = cgm_df[cgm_df['Glucose reading'].notnull()]
    df = df.sort_values('Actual time')
    df['diff_next_sample'] = df['Actual time'].diff().dt.days
    df['tag_no'] = df['diff_next_sample'] > 1
    df['tag_no'] = df['tag_no'].cumsum()
    #df['month'] = df['Actual time'].dt.to_period('M').astype(str)
    df['Tag No'] = df.tag_no.apply(lambda x: f'Tag_No - {x+1}')
    df =df.fillna(0)

    return df

def food_preprocess_data(cgm_df):
    format = '%d-%m-%Y %H:%M%p'
    try:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'], format=format)
    except:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'])

    cgm_df = cgm_df[['Actual time', 'Glucose reading', 'Event type / Notif type', 'Event name']]
    cgm_df['Event type / Notif type'] = cgm_df['Event type / Notif type'].shift(-1)
    cgm_df['Event name'] = cgm_df['Event name'].shift(-1)
    cgm_df = cgm_df[['Actual time', 'Glucose reading', 'Event type / Notif type', 'Event name']]
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
    glucose = glucose[['Actual time', 'Glucose reading', 'Event type / Notif type', 'Event name']]
    import pandas as pd
    glucose['Actual time'] = pd.to_datetime(glucose['Actual time'])
    glucose['Day']=glucose['Actual time'].dt.date
    glucose['Time']=glucose['Actual time'].dt.hour
    glucose['weekDay'] = glucose['Actual time'].dt.day_name()
    glucose=glucose.reset_index(drop=True)
    
    date1=str(glucose["Actual time"].min().date())
    date2 = str(glucose["Actual time"].max().date()) 
    date3 = str((glucose["Actual time"].max() - glucose["Actual time"].min()).days)
    
    #st.dataframe(data=glucose, width=1024, height=768)

   # Glucose Matrices calculation
    
    food_d=glucose[glucose['Event type / Notif type'].notnull()] 
    
 
       
    
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
    
    spike_data=stats_df[['Spike No', 'time_to_reach_top',
            'Max Glucose', 'start_time', 'end_time']]
    spike_data=spike_data.rename(columns={'time_to_reach_top': 'Time spent in spike'})
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
    grop2 = len(glucose[glucose['Glucose reading'].between(111,160)])
    grop3 = len(glucose[glucose['Glucose reading'].between(70,110)])
    grop4 = len(glucose[glucose['Glucose reading'].between(61,69)])
    grop5 = len(glucose[glucose['Glucose reading'].between(30,60)])
    
    glucose['HbA1c'] = (46.7 + glucose['Glucose reading']) / 28.7
    glucose['day/night'] = glucose['Time'].apply(lambda x: 'night' if 0 <= x <= 8 or 20 <= x <= 23 else 'day')
    comp_glucose =glucose['Glucose reading']
    YY = []
    for k in range(len(comp_glucose)):
            if np.all(comp_glucose[k]>160):
                  YY.append('Very high (>161mg/dL)')
            elif np.all(110<comp_glucose[k]<=160):
                 YY.append('High (111-160mg/dL)')
            elif np.all(70<comp_glucose[k]<=110):
                YY.append('Normal (70-110mg/dL)')
            elif np.all(60<comp_glucose[k]<=70):
                YY.append('Low (60-70mg/dL)')
            else:
                YY.append('Very low (<70mg/dL)')
    glucose['group_status'] =YY      
    #import plotly.graph_objects as go
    #st.dataframe(data=glucose, width=1024, height=768)
    unique_very_high = glucose[glucose['group_status'] == 'Very high (>161mg/dL)']['Day'].nunique()
    unique_very_low = glucose[glucose['group_status'] == 'Very low (<70mg/dL)']['Day'].nunique()
    unique_high = (glucose[glucose['group_status'] == 'High (111-160mg/dL)']['Day'].nunique())-1
    unique_low = glucose[glucose['group_status'] == 'Low (60-70mg/dL)']['Day'].nunique()
    
    zy=[]
    for j in range(len(comp_glucose)):
            if np.all(70<comp_glucose[j]<=110):
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
    
    
    #radar data----------------
    def normalize(value, minimum, maximum):
        return (value - minimum) / (maximum - minimum) * (5 - 0) + 0

    
    
    peak_day_AVG=peak_day['Spikes'].mean()
    spikeperday_score=normalize(peak_day_AVG,4,7)
    
    timespentspike=spike_data['Time spent in spike'].mean()
    timeinspike_score=normalize(timespentspike,20,100)
    
    avg_glucose_score=normalize(avg_glucose,70,160)
    day_glucose_score=normalize(day_averageG,70,160)
    night_glucose_score=normalize(night_averageG, 70,160)
    
    import pandas as pd
    df = pd.DataFrame(dict(
        score=[ TINR_score, average_time_spent_in_peaks_score,spikeperday_score, avg_glucose_score, day_glucose_score, night_glucose_score],
        metric=['Percent time inside range 70-110', 'Average Time in spike', 'spikes in a day score' ,'Highest glucose reading per day', 'day glucose reading score', 'night glucose reading score'
                ]))





    #Redarfig = px.scatter(df, y='score', x='metric',  size_max=60)
    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    

    st.subheader("Metrices Score :")
    #st.plotly_chart(Redarfig, use_container_width=True,theme="streamlit")
    st.plotly_chart(Redarfig, use_container_width=True)
    st.markdown("""---""")
    
    
    Labels= ['Very high (>161 mg/dL)', 'High(111-160 mg/dL)', 'Normal(70-110 mg/dL)', 'Low(60-70 mg/dL)', 'Very Low(<60 mg/dL)']
    sizes= [grop1, grop2, grop3, grop4, grop5]
    #explode = (0.4, 0, 0, 0, 0.4)
    
    colval, colpie = st.columns([1,3])
       
    figz= px.pie(sizes, values=sizes, names = Labels, color=Labels, title = 'Percentage time spent in glucose reading ranges',
                 color_discrete_map={
                'Very high (>161 mg/dL)': "red",
                'Normal(70-110 mg/dL)': "#49B55E",
                'Very Low(<60 mg/dL)': "#6495ED",
                'Low(60-70 mg/dL)': "#191970",
                'High(111-160 mg/dL)': "#FBAD5E"})
    figz.update_traces(textposition= 'inside', textinfo= 'percent+label', hole=.5)
    figz.update_layout(title_font_size = 42, autosize=False, width=500, height=500, margin=dict(l=10, r=10, t=20, b=5) )
    colpie.plotly_chart(figz)
    #source = data.seattle_weather()
    colval.markdown('Average glucose')
    colval.subheader(avg_glucose)
    colval.markdown('Average at night')
    colval.subheader(night_averageG)
    colval.markdown('Average at night')
    colval.subheader(day_averageG)
    colval.markdown('''Max glucose reading 
                  in time period:''')
    colval.subheader(high_glucose)            
    colval.markdown('''Min glucose reading 
                  in time period:''')
    colval.subheader(low_glucose)               
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
    
    with st.expander(" **Average glucose readings in range over time frame**"):
        pg1, pg2, pg3, pg4, pg5 = st.columns(5)
         
        pg1.metric('Very High(>161mg/dL)', valg1)
        pg1.markdown('''Number of days in 
                     very high range''')
        pg1.markdown(unique_very_high)
        
        pg2.metric('High(111-160mg/dL)', valg2)
        pg2.markdown('''Number of days in 
                     high range''')
        pg2.markdown(unique_high)
        
        pg3.metric('Normal(70-110mg/dL)', valg3)
        pg3.markdown('''Number of days in 
                     Normal range''')
        pg3.markdown(unique_nor) 
            
        
        pg4.metric('Low(60-70mg/dL)', valg4)
        pg4.markdown('''Number of days in 
                     low range''' )
        pg4.markdown(unique_low)
        
        pg5.metric('Very Low(<60mg/dL)', valg5 )
        pg5.markdown('''Number of days in 
                     very low range''')
        pg5.markdown(unique_very_low)

        st.write('Number of days spent with Hyperglycemia:', unique_hyper  )
        st.write('Number of days spent with Hypoglycemic:', unique_hypo )
    
        
    
    import base64
    
    tabh, tabwk, tabz  =st.tabs(["Glucose Analysis",  "Day/Night Analysis","Group Pattern Data" ])
    with tabh:
         
         fig_group = px.scatter(glucose, x="Actual time", y="Glucose reading", color="group_status", 
                     color_discrete_map={
                        'Very high (>161mg/dL)': "red",
                        'High (111-160mg/dL)': "#FBAD5E",
                        'Normal (70-110mg/dL)': "#49B55E",
                        'Low (60-70mg/dL)': "#0D2A63",
                        'Very Low (<60mg/dL)': "skyblue"},
                        title="CGM Group Analysis"
                        )
         fig_group.update_yaxes(title_text='Glucose value (mg/dL)')
         st.write(fig_group, use_container_width=True )  
         
         chartC = alt.Chart(data= glucose, title = "Average at specific time point for 24 hours").mark_bar().encode(
          x='hours(Actual time):O',
           y=alt.Y('average(Glucose reading)', title='Glucose reading(mg/dL)')
             )
         rule = alt.Chart(glucose).mark_rule(color='red').encode(
           y='mean(Glucose reading):Q')
    
         chart = (chartC + rule).properties(width=600)
         st.altair_chart(chart, theme="streamlit", use_container_width=True)

         bar_rounded1 = alt.Chart(glucose, title='Average of Glucose reading by day ').mark_bar().encode(
                x='monthdate(Day):O',
                y='average(Glucose reading)'
                ).properties(width=50)
         rule7 = alt.Chart(glucose).mark_rule(color='red').encode(
           y='average(Glucose reading):Q')
    
         chartd = (bar_rounded1).properties(width=600)      
         st.altair_chart(chartd, theme=None, use_container_width=True )
    with tabz:
        
         
         # table interactive
         gener_group = st.radio("Please select a range",('Very high (>161 mg/dL)', 'High (111 - 160 mg/dL)', 'Normal (70 - 110 mg/dL)', 'Low (60 - 70 mg/dL)', 'Very low (<70 mg/dL)'), horizontal=True) 
         
         if gener_group == 'Very high (>161 mg/dL)':
             glucose_group = glucose2.loc[glucose2['group_status']=='Very high (>161mg/dL)']
             st.download_button("Press to Download table",
             convert_df(glucose_group),
             "cgm_stats_Very_high.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)

         elif gener_group == 'High (111 - 160 mg/dL)':
             glucose_group = glucose2.loc[glucose2['group_status']=='High (111-160mg/dL)']
             st.download_button("Download table",
             convert_df(glucose_group),
             "cgm_stats_High.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)            
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='high'], width=1024, height=768)
             
         elif gener_group == 'Normal (70 - 110 mg/dL)':
             glucose_group = glucose2.loc[glucose2['group_status']=='Normal (70-110mg/dL)']
             st.download_button("Download table",
             convert_df(glucose_group),
             "cgm_stats_Normal.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768)        
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='Normal'], width=1024, height=768)
             
         elif gener_group == 'Low (60 - 70 mg/dL)':
             glucose_group = glucose2.loc[glucose2['group_status']=='Low (60-70mg/dL)']
             st.download_button("Download table",
             convert_df(glucose_group),
             "cgm_stats_Low.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768) 
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='low'], width=1024, height=768) 
             
         else :
             glucose_group = glucose2.loc[glucose2['group_status']=='Very low (<70mg/dL)']
             st.download_button("Download table",
             convert_df(glucose_group),
             "cgm_stats_High.csv",
             "text/csv",
             key='download_csv')
             st.dataframe(data= glucose_group, width=1024, height=768) 
             #st.dataframe(data= glucose2.loc[glucose2['group_status']=='very_low'], width=1024, height=768)

                 
    with tabwk:
        
  
    #candlestick chart for glucose
         scale1 = alt.Scale(
         domain=['day','night'],
         range=[ "#49B55E", "#191970"],)
         color1 = alt.Color('day/night:N', title='Key',scale=scale1)
    
           
        
        
         bar_rounded2 = alt.Chart(glucose,title='Glucose values (mg/dL): day (8 am - 8 pm) vs night (8 pm - 8 am)').mark_line(
            ).encode(
                x='monthdate(Day):O',
                y=alt.Y('average(Glucose reading)', title='Average glucose values (mg/dL)'),
                color=color1,
                ).properties(width=50)
              
         st.altair_chart(bar_rounded2 , theme=None, use_container_width=True ) 

    
    tab1, tab2 = st.tabs(["HbA1", "Spike"])
    with tab1:
         gm2, gm3, gm4  = st.columns((1, 1, 1))
         gm2.metric("American Diabetes Association-HbA1c", ADA_HbA1c )
         gm3.metric("Percent time inside range 70-110mg/dL", pTIR )
         gm4.metric("Average time to Peak (Min)", ATtoFloorfrom_Peak )
         
         gm2.metric("Average time to floor (Min)", ATtoTopfrom_base )
         gm3.metric("Average BG", average_bg)
         #gm1.markdown('**Glucose management index-HbA1c**')
         #gm1.subheader(GMI_HbA1c)
         #gm2.markdown('**American Diabetes Association-HbA1c**')
         #gm2.subheader(ADA_HbA1c)
         #gm3.markdown('**Average Daily Risk Range**')
         #gm3.subheader(Adrr)
         #gm4.markdown('**Percent time inside range 70-150**')
         #gm4.subheader(pTIR)
         
                      
                     
    with tab2:
        st.header("Spikes Average Calculation")
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
            fig1.update_layout(title_text="Spikes (>110 mg/dL) over time", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig1)
            
            
        with st.expander("**Crashes (<70 mg/dL) over time**"):
            fig_crash = px.bar(crash_day, x='Date', y= 'crash')
            fig_crash.update_traces(marker_color='#191970')
            fig_crash.update_layout(title_text="Crashes in a Day", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig_crash)
            st.write(''' crashes of glucose reading less than 70 mg/dL.''')  
            
        colorMR=[]    
        for h in stats_df['Max Glucose']:
            if h > 160:
                colorMR.append('(>160mg/dL)')
            elif h > 130:
                colorMR.append('(130-160mg/dL)')
            else:
                colorMR.append('(<130mg/dL)')    
        figMR = px.bar(spike_data, x="Spike No", y="Max Glucose",color= colorMR, text=spike_data['Time spent in spike'],hover_data=['start_time','end_time'])
        figMR.update_traces(textposition='outside')
        figMR.update_yaxes(title_text='Glucose value (mg/dL)')
        
        
            
        figMR.update_layout(title_text="Maximum Glucose in spike", title_x=0, margin=dict(l=0, r=5, b=5, t=50))
        st.plotly_chart(figMR)
        
            
            

    
    
    
    

    
    # Line chart ploting on dash board

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
        local_df = pd.read_csv('cricketer_data.csv')
        open_dash(local_df)





