#Visualization of relationship between ​chronic diseases and preventions in 500 US Cities​
CSE6242 Data and Visual Analytics Project 

## DESCRIPTION
This package is used for analysis and interactive visualization to show the most prevalent diseases in the 500 largest cities in the US, 
the factors that most contribute to those poor health outcomes and their prevalence within the cities with data provided by the Centers for Disease Control and Prevention (CDC).
The packages contains two part. 
The first part is implemented using Python. It takes the raw data input, perform data cleaning and modeling, generate the results and a pre-defined formatted file for further processing.
The second part takes use of Tableau to provide an interactive visualization interface for people to understand the results we found from analysis. 

## INSTALLATION
1. Download the code base to local. 
2. Install python 3 with numpy, pandas and scikit-learn libraries.
3A. IMPORTANT: RUNNING THE ANALYSIS ON THE FULL DATASET WILL TAKE MANY (10-20) hours. We have included an extract of the data for faster testing. Simply copy the included file below:
    500_Cities__Local_Data_for_Better_Health__2018_release_sample.csv
    AS
    500_Cities__Local_Data_for_Better_Health__2018_release.csv
    Alternatively:
    Download raw data from below link and put it into the project code base folder. 
    https://chronicdata.cdc.gov/500-Cities/500-Cities-Local-Data-for-Better-Health-2018-relea/6vp6-wxuq

## EXECUTION 
For analysis, run main.py file, three modes has been provided in the program. Follow the instruction from the console, it will ask you to enter the mode you want to use
1. Perform top feature selection Analysis.
This will select the top n factors that contribute to each of the health outcomes of all the states in US. 
2. Generate output file for visualizatio - named visual_input_tract_with_city.csv
This is to generate the file used by our following visualization part. It generate the top 5 contributing factors of all the state and apply it to city level as well as provide the prevalence information of each health outcome and factors.
3. Run multiple models comparision analysis.
In this mode, you can select country level or a particular state to train the data with multiple regression models including linear regression, Ridge regression, Lasso regression and Support vector regression and compare the prediction accuracy and model information such as factor coefficients across different models.

## STEPS TO GENERATE INPUT DATA FOR TABLEAU
The final visualization is available for viewing at:
https://public.tableau.com/profile/myles.lefkovitz#!/vizhome/500_Cities_Visualizations/Story
#############
The  tableau workbook and the input data file for the visualization are also included in this folder
    500_Cities_Visualizations.twbx (Tableau workbook)
    visualization_input.csv (input file for Tableau workbook)
To re-generate the input file, follow the below steps (note, the anlysis on the full data takes roughly 15 hours):

1. Generate the output file from mode 2 - visual_input_tract_with_city.csv
2. Copy all the data from that file to data_prep.xlsx file provided, which maps the features with more meaningful descriptions and add the geographic location of each city and state to the file.  Remember to exclude the heading when you copy.
3. Save the new datasheet as a csv called visualization_input.csv. This will overwrite the existing visualization_input file.
4. Open the included Tableau file in Tableau, select Data tab and select the new data input file.


FEATURE EXPLAINATIONS:

OUTCOMES:
ARTHRITIS: Arthritis among adults aged≥18 years
CASTHMA: Current asthma among adults aged≥18 years
BPHIGH: High blood pressure among adults aged≥18 years
CANCER: Cancer (excluding skin cancer) among adults aged≥18 years
HIGHCHOL: High cholesterol among adults aged≥18 years who have been screened in the past 5 years
KIDNEY: Chronic kidney disease among adults aged≥18 years
COPD: Chronic obstructive pulmonary disease among adults aged≥18 years 
CHD: Coronary heart disease among adults aged≥18 years
DIABETES: Diagnosed diabetes among adults aged≥18 years
MHLTH: Mental health not good for≥14 days among adults aged≥18 years
PHLTH: Physical health not good for≥14 days among adults aged≥18 years
TEETHLOST: All teeth lost among adults aged≥65 years
STROKE: Stroke among adults aged≥18 years

PREVENTIONS:
ACCESS2: Current lack of health insurance among adults aged 18–64 years
CHECKUP: Visits to doctor for routine checkup within the past year among adults aged≥18 years
DENTAL: Visits to dentist or dental clinic among adults aged ≥18 years
BPMED: Taking medicine for high blood pressure control among adults aged≥18 years with high blood pressure
CHOLSCREEN: Cholesterol screening among adults aged≥18 years
MAMMOUSE: Mammography use among women aged 50–74 years
PAPTEST: Papanicolaou smear use among adult women aged 21–65 years
COLON_SCREEN: Fecal occult blood test, sigmoidoscopy, or colonoscopy among adults aged 50–75 years
COREW: Older women adults aged≥65 years who are up to date on a core set of clinical preventive services
COREM: Older men adults aged≥65 years who are up to date on a core set of clinical preventive services

UNHEALTHY BEHAVIORS:
BINGE: Binge drinking among adults aged≥18 years
CSMOKING: Current smoking among adults aged≥18 years
LPA: No leisure-time physical activity among adults aged≥18 years
OBESITY: Obesity among adults aged≥18 years
SLEEP: Sleeping less than 7 hours among adults aged ≥18 years
