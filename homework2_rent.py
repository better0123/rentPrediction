__author__ = "EMily"

import warnings
warnings.filterwarnings("ignore")
import sys
if sys.version_info[0] < 3:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import io

def process():
    '''
    read the data from url
    return data frame 
    '''
    target_url = "https://ndownloader.figshare.com/files/7586326"
    urlData = requests.get(target_url).content
    rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
    #features with rent associalted information, needed to be removed
    remove_col = ['uf17a','uf29','hflag18','uf27','hflag4','uf26','uf28','uf27','uf30']
    df = rawData.drop(remove_col,inplace=False,axis=1)
    df = df[df['uf17'] != 99999] #remove NA in uf17 (dependent variable)
    #########special process#########
    to_be_binary_dic = {'recid' : {1:'OCCUPIED'},'uf1_1' : {1:'Missing bricks, siding, or other outside wall material'},'uf1_2' : {1:'Sloping or bulging outside walls'},'uf1_3' : {1:'Major cracks in outside walls'},'uf1_4' : {1:'Loose or hanging cornice, roofing, or other material'},'uf1_5' : {1:'None of these problems with walls'},'uf1_6' : {1:'Unable to observe walls'},'uf1_7' : {1:'Broken or missing windows'},'uf1_8' : {1:'Rotten/loose window frames/sashes'},'uf1_9' : {1:'Boarded-up windows'},'uf1_10' : {1:'None of these problems with windows'},'uf1_11' : {1:'Unable to observe windows'},'uf1_12' : {1:'Loose, broken, or missing stair railings'},'uf1_13' : {1:'Loose, broken, or missing steps'},'uf1_14' : {1:'None of these problems with stairways'},'uf1_15' : {1:'No interior steps or stairways'},'uf1_16' : {1:'No exterior steps or stairways'},'uf1_35' : {1:'Unable to observe stairways'},'uf1_17' : {1:'Sagging or sloping floors'},'uf1_18' : {1:'Slanted or shifted doorsills or door frames'},'uf1_19' : {1:'Deep wear in floors causing depressions'},'uf1_20' : {1:'Holes or missing flooring'},'uf1_21' : {1:'None of these problems with floors'},'uf1_22' : {1:'Unable to observe floors'},'sc115' : {1:'Owned or being bought'}}

    #turn features in to_be_binary_dic into binary features
    for key in to_be_binary_dic.keys():
        possible_values = to_be_binary_dic[key].keys()
        df[key][~(df[key].isin(possible_values))] = 0
    #########binary feature names#########
    binary_dic = {'hhr2' : {1:'Male', 2:'Female'},'sc54' : {1:'Yes, first occupants', 2:'No, previously occupied'},'sc121' : {1:'Yes', 2:'No'},'sc140' : {1:'Yes', 2:'No'},'sc143' : {1:'Yes', 2:'No'},'sc147' : {1:'Yes', 2:'No'},'sc149' : {1:'Yes', 2:'No'},'sc173' : {1:'Yes', 2:'No'},'sc171' : {1:'Yes', 2:'No'},'sc117' : {1:'Yes', 2:'No'},'sc118' : {1:'Yes', 2:'No'},'sc174' : {1:'Yes', 2:'No'},'sc541' : {1:'Yes', 2:'No'},'sc184' : {1:'Yes', 2:'No'},'sc542' : {1:'Yes', 2:'No'},'sc543' : {1:'Yes', 2:'No'},'sc544' : {1:'Yes', 2:'No'},'sc198' : {1:'Yes', 2:'No'},'sc187' : {1:'Yes', 2:'No'},'sc188' : {1:'Yes', 2:'No'},'sc190' : {1:'Yes', 2:'No'},'sc191' : {1:'Yes', 2:'No'},'sc194' : {1:'Yes', 2:'No'},'sc548' : {1:'Yes', 2:'No'},'sc549' : {1:'Yes', 2:'No'},'sc550' : {1:'Yes', 2:'No'},'sc551' : {1:'Yes', 2:'No'},'sc575' : {1:'Yes', 2:'No'},'sc560' : {1:'Yes', 2:'No'},'sc24' : {1:'Yes', 2:'No'},'sc185' : {0:'Yes', 1:'No'},'sc192' : {0:'Yes', 1:'No'},'sc193' : {2:'Yes', 3:'No'}, 'sc125' : {0:'Don\'t know'},'uf52h_h' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_a' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_b' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_c' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_d' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_e' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_f' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'uf52h_g' : {1:'Any amount allocated ', 0:'All amounts not allocated'},'flg_hs1' : {1:'Allocated', 0:'Not allocated'},'flg_rc1' : {1:'Allocated', 0:'Not allocated'},'flg_sx1' : {1:'Allocated', 0:'Not allocated'},'rec21' : {1:'Not dilapidated' , 2:'Dilapidated'},'uf19' : {1:'Found', 2:'Not found'},'recid' : {1:'OCCUPIED', 0: 'no'},'uf1_1' : {1:'Missing bricks, siding, or other outside wall material', 0: 'no'},'uf1_2' : {1:'Sloping or bulging outside walls', 0: 'no'},'uf1_3' : {1:'Major cracks in outside walls', 0: 'no'},'uf1_4' : {1:'Loose or hanging cornice, roofing, or other material', 0: 'no'},'uf1_5' : {1:'None of these problems with walls', 0: 'no'},'uf1_6' : {1:'Unable to observe walls', 0: 'no'},'uf1_7' : {1:'Broken or missing windows', 0: 'no'},'uf1_8' : {1:'Rotten/loose window frames/sashes', 0: 'no'},'uf1_9' : {1:'Boarded-up windows', 0: 'no'},'uf1_10' : {1:'None of these problems with windows', 0: 'no'},'uf1_11' : {1:'Unable to observe windows', 0: 'no'},'uf1_12' : {1:'Loose, broken, or missing stair railings'},'uf1_13' : {1:'Loose, broken, or missing steps', 0: 'no'},'uf1_14' : {1:'None of these problems with stairways', 0: 'no'},'uf1_15' : {1:'No interior steps or stairways', 0: 'no'},'uf1_16' : {1:'No exterior steps or stairways', 0: 'no'},'uf1_35' : {1:'Unable to observe stairways', 0: 'no'},'uf1_17' : {1:'Sagging or sloping floors', 0: 'no'},'uf1_18' : {1:'Slanted or shifted doorsills or door frames', 0: 'no'},'uf1_19' : {1:'Deep wear in floors causing depressions', 0: 'no'},'uf1_20' : {1:'Holes or missing flooring', 0: 'no'},'uf1_21' : {1:'None of these problems with floors', 0: 'no'},'uf1_22' : {1:'Unable to observe floors', 0: 'no'},'sc115' : {1:'Owned or being bought', 0: 'no'}}
    #########replace no reply with np.nan for the binary variables##########
    binary_keys = binary_dic.keys()
    df_binary = df[binary_keys]
    for key in binary_keys:
        possible_values = binary_dic[key].keys() #possible_values : list of known-value
        df_binary[key][~(df_binary[key].isin (possible_values))] = np.nan #replace anything outside known-value with NaN

    #########process categorical data#########
    categorical_dic = {'boro' : {1:'Bronx',2:'Brooklyn',3:'Manhattan',4:'Queens',5:'Staten_Island'},'sc23' : {1:'Dilapidated', 2:'Sound', 3:'Deteriorating'},'sc36' : {1:'Accessible', 2:'Inaccessible', 3:'Unable to observe building entrance'},'sc37' : {1:'Accessible', 2:'Inaccessible', 3:'Unable to observe elevator', 4:'No elevator'},'sc38' : {1:'Accessible', 2:'Inaccessible', 3:'Unable to observe residential unit entrance'},'hhr5' : {1:'No',2:'Puerto Rican',3:'Dominican',4:'Cuban',5:'South/Central American',6:'Mexican, Mexican-American, Chicano',7:'Other Spanish/Hispanic'},'sc51' : {1:'Always lived in this unit', 2:'Other unit in same building', 3:'Bronx', 4:'Brooklyn', 5:'Manhattan', 6:'Queens', 7:'Staten Island', 8:'NY, NJ, Connecticut', 9:'Other state', 10:'Puerto Rico', 11:'Dominican Republic', 12:'Caribbean (other than Puerto Rico or Dom. Rep.) 13:Mexico', 14:'Central America, South America 15:Canada', 16:'Europe', 17:'Russia/Successor States to Soviet Union 18:China, Hong Kong, Taiwan', 19:'Korea', 20:'India', 21:'Pakistan, Bangladesh', 22:'Philippines', 23:'Southeast Asia (Burma, Cambodia, Laos, Malaysia, Singapore, Thailand, Vietnam) 24:Other Asia', 25:'Africa', 26:'All other countries'},'sc52' : {1:'2012 or later', 2:'2009-2011', 3:'2006-2008', 4:'2003-2005', 5:'2000-2002', 6:'1995-1999', 7:'1990-1994', 8:'1985-1989', 9:'1980-1984', 10:'1970-1979', 11:'1969 or earlier'},'sc53' : {1:'Yes, on or after July 1 in 1971', 2:'No, before July 1, 1971', 9:'Not applicable (did not move in 1971)'},'sc110' : {1:'Job transfer/new job ', 2:'Retirement', 3:'Looking for work', 4:'Commuting reasons', 5:'To attend school', 6:'Other financial/employment reason ', 7:'Needed larger house or apartment ', 8:'Widowed', 9:'Separated/divorced', 10:'Newly married', 11:'Moved to be with or closer to relatives', 12:'Family decreased (except widowed/separated/divorced) ', 13:'Wanted to establish separate household', 14:'Other family reason', 15:'Neighborhood overcrowded', 16:'Change in racial or ethnic composition of neighborhood ', 17:'Wanted this neighborhood/better neighborhood services ', 18:'Crime or safety concerns', 19:'Other neighborhood reason', 20:'Wanted to own residence', 21:'Wanted to rent residence', 22:'Wanted less expensive residence/difficulty paying rent or mortgage', 23:'Wanted better quality residence', 24:'Evicted', 25:'Poor building condition/services', 26:'Harassment by landlord', 27:'Needed housing accessible for persons with mobility impairments ', 28:'Other housing reason', 29:'Displaced by urban renewal, highway construction, or other public activity ', 30:'Displaced by private action (other than eviction) ', 31:'Schools', 32:'Natural disaster/fire', 33:'Any other reason'},'sc111' : {7:'New York City', 9:'U.S., outside New York City', 10:'Puerto Rico', 11:'Dominican Republic', 12:'Caribbean (other than Puerto Rico or Dom. Rep.) 13:Mexico', 14:'Central America, South America', 15:'Canada', 16:'Europe', 17:'Russia/Successor States to Soviet Union ', 18:'China, Hong Kong, Taiwan', 19:'Korea', 20:'India', 21:'Pakistan, Bangladesh', 22:'Philippines', 23:'Southeast Asia (Burma, Cambodia, Laos, Malaysia, Singapore, Thailand, Vietnam) ', 24:'Other Asia', 25:'Africa', 26:'All other countries'},'sc112' : {7:'New York City', 9:'U.S., outside New York City', 10:'Puerto Rico', 11:'Dominican Republic', 12:'Caribbean (other than Puerto Rico or Dom. Rep.) 13:Mexico', 14:'Central America, South America', 15:'Canada', 16:'Europe', 17:'Russia/Successor States to Soviet Union ', 18:'China, Hong Kong, Taiwan', 19:'Korea', 20:'India', 21:'Pakistan, Bangladesh', 22:'Philippines', 23:'Southeast Asia (Burma, Cambodia, Laos, Malaysia, Singapore, Thailand, Vietnam) ', 24:'Other Asia', 25:'Africa', 26:'All other countries'},'sc113' : {7:'New York City', 9:'U.S., outside New York City', 10:'Puerto Rico', 11:'Dominican Republic', 12:'Caribbean (other than Puerto Rico or Dom. Rep.) 13:Mexico', 14:'Central America, South America', 15:'Canada', 16:'Europe', 17:'Russia/Successor States to Soviet Union ', 18:'China, Hong Kong, Taiwan', 19:'Korea', 20:'India', 21:'Pakistan, Bangladesh', 22:'Philippines', 23:'Southeast Asia (Burma, Cambodia, Laos, Malaysia, Singapore, Thailand, Vietnam) ', 24:'Other Asia', 25:'Africa', 26:'All other countries'},'sc114' : {1:'No', 2:'Yes, a condominium' , 3:'Yes, a cooperative'},'sc120' : {1:'Owned and occupied by another household ', 2:'Rented by reference person', 3:'Rented by another household', 4:'Never previously occupied'},'sc116' : {2:'Pay cash rent', 3:'Occupy rent free', 9:'Not applicable (owner occupied)'},'sc127' : {1:'Mortgage, home equity, or similar loan', 2:'Owned free and clear', 9: 'Not applicable (renter occupied or occupied rent free)'},'uf9' : {1:'Less than $100', 2:'$100-$199', 3:'$200-$299', 4:'$300-$399', 5:'$400-$499', 6:'$500-$599', 7:'$600-$699', 8:'$700-$799', 9:'$800-$899', 10:'$900-$999', 11:'$1,000-$1,249', 12:'$1,250-$1,499', 13:'$1,500-$1,749', 14:'$1,750-$1,999', 15:'$2,000-$2,249', 16:'$2,250-$2,499', 17:'$2,500-$2,999', 18:'$3,000 or more'},'sc141' : {1:'Yes', 2:'No, included in mortgage or loan payment', 3:'No insurance'},'sc144' : {1:'Yes', 2:'No, included in mortgage or loan payment', 3:'No, included in condominium or maintenance fee'},'uf10' : {1:'Less than $100', 2:'$100-$199', 3:'$200-$299', 4:'$300-$399', 5:'$400-$499', 6:'$500-$599', 7:'$600-$699', 8:'$700-$799', 9:'$800-$899', 10:'$900-$999', 11:'$1,000-$1,249', 12:'$1,250-$1,499', 13:'$1,500-$1,749', 14:'$1,750-$1,999', 15:'$2,000-$2,499', 16:'$2,500-$2,999', 17:'$3,000-$3,499', 18:'$3,500-$3,999', 19:'$4,000-$4,499', 20:'$4,500-$4,999', 21:'$5,000-$5,499', 22:'$5,500-$5,999', 23:'$6,000-$6,499', 24:'$6,500-$6,999', 25:'$7,000-$7,499', 26:'$7,500-$7,999', 27:'$8,000-$8,499', 28:'$8,500-$8,999', 29:'$9,000-$9,999', 30:'$10,000 or more'},'uf48' : {1:'1 unit without business', 2:'1 unit with business', 3:'2 units without business', 4:'2 units with business', 5:'3 units', 6:'4 units', 7:'5 units', 8:'6 to 9 units', 9:'10 to 12 units', 10:'13 to 19 units', 11:'20 to 49 units', 12:'50 to 99 units', 13:'100 units or more'},'uf11' : {1:'1 to 2 stories ', 2:'3 stories', 3:'4 stories', 4:'5 stories', 5:'6 to 10 stories ', 6:'11 to 20 stories ', 7:'21 stories or more'},'sc150' : {1:'1 room',2:'2 room',3:'3 room',4:'4 room',5:'5 room',6:'6 room',7:'7 room',8:'8 room'},'sc151' : {1:'No bedroom', 2:'1 bedroom', 3:'2 bedrooms', 4:'3 bedrooms', 5:'4 bedrooms', 6:'5 bedrooms', 7:'6 bedrooms', 8:'7 bedrooms', 9:'8 bedrooms or more'},'sc152' : {0:'Yes, complete plumbing facilities', 1:'No, has some but not all facilities in this apartment (house)', 2:'No plumbing facilities in this apartment (house)'},'sc153' : {3:'For the exclusive use of this household', 4:'Also for use by another household'},'sc154' : {1:'Yes', 2:'No', 3:'No toilet in this apartment (house)'},'sc155' : {0:'Yes, has complete kitchen facilities', 1:'No, has some but not all facilities in this apartment (house) ', 2:'No kitchen facilities in this apartment (house), but facilities available in building ', 3:'No kitchen facilities in this building'},'sc156' : {4:'For exclusive use of this household', 5:'Also for use by another household'},'sc157' : {1:'Yes, all functioning', 2:'No, one or more is not working at all'},'sc158' : {1:'Fuel oil', 2:'Utility gas', 3:'Electricity', 4:'Other fuel (including CON ED steam)'},'sc159' : {1:'Yes', 2:'Yes, but combined with gas', 3:'No, included in rent, condominium, or other fee'},'sc161' : {1:'Yes', 2:'No, included in rent, condominium or other fee', 3:'No, gas not used'},'sc164' : {1:'Yes', 2:'No, included in rent, condominium or other fee or no charge'},'sc166' : {1:'Yes', 2:'No, included in rent, condominium or other fee ', 3:'No, these fuels not used'},'sc181' : {1:'Less than 1 year', 2:'1 Year', 3:'More than 1 but less than 2 years', 4:'2 years', 5:'More than 2 years', 6:'No lease'},'sc186' : {2:'One time', 3:'Two times ', 4:'Three times ', 5:'Four or more times'},'sc197' : {1:'Yes, central air conditioning', 2:'Yes, one or more window air conditioners ', 3:'No'},'sc571' : {1:'None', 2:'1 to 5', 3:'6 to 19', 4:'20 or more'},'sc189' : {1:'Regularly', 2:'Only when needed ', 3:'Irregularly', 4:'Not at all'},'sc196' : {1:'Excellent' , 2:'Good' , 3:'Fair', 4:'Poor'},'sc199' : {1:'Daily' , 2:'Weekly' , 3:'Monthly' , 4:'A few times' , 5:'Never'},'sc570' : {0:'None', 1:'1 person', 2:'2 person', 3:'3 person', 4:'4 person', 5:'5 person', 6:'6 person', 7:'7 person', 8:'8 person', 9:'9 person', 10:'10 person', 11:'11 person', 12:'12 person', 13:'13 person', 14:'14 person', 15:'15 person'},'sc574' : {1:'Excellent ', 2:'Very Good ', 3:'Good' , 4:'Fair', 5:'Poor'},'new_csr' : {1:'Owner occupied conventional ', 2:'Owner occupied private cooperative', 5:'Public housing', 12:'Owner occupied condo ', 20:'Article 4 or 5 building', 21:'HUD regulated', 22:'Loft Board regulated building ', 23:'Municipal Loan Program ', 30:'Stabilized pre 1947 ', 31:'Stabilized post 1947', 80:'Other rental', 85:'Mitchell Lama rental ', 86:'Mitchell Lama cooperative ', 87:'Mitchell Lama type cooperative ', 90:'Controlled', 95:'In Rem'},'rec15' : {1:'Old law tenement (built pre 1901) ', 2:'New law tenement (built 1901-1929) ', 3:'Multiple built after 1929 (including public housing)', 4:'Apartment hotel (built before 1929)', 5:'One or two family converted to apartments ', 6:'Commercial building altered to apartments ', 7:'Tenement building used for single-room occupancy ', 8:'One or two family converted to rooming house ', 9:'Miscellaneous Class B Structure'},'sc26' : {1:'Regular', 2:'Cooperative' , 3:'Condominium', 12:'Public Housing ', 13:'New Construction ', 15:'In Rem', 16:'Old Construction'},'uf23' : {1:'2000 or later ', 2:'1990 to 1999 ', 3:'1980 to 1989 ', 4:'1974 to 1979 ', 5:'1960 to 1973 ', 6:'1947 to 1959 ', 7:'1930 to 1946 ', 8:'1920 to 1929 ', 9:'1901 to 1919', 10:'1900 and earlier'},'sc27' : {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10', 11:'11', 12:'12', 13:'13', 14:'14'},'rec1' : {1:'With no other household members', 2:'With children under 18 only', 3:'With no children under 18', 4:'With other adults and children under 18', 6:'With no other household members 7:With children under 18 only', 8:'With no children under 18', 9:'With other adults and children under 18', 11:'With no other household members ', 12:'With children under 18 only', 13:'With no children under 18', 14:'With other adults and children under 18'},'uf46' : {1:'Relatives present only', 2:'Nonrelatives present'},'rec4' : {1:'White only, not-Hispanic ', 2:'Black only, not-Hispanic', 3:'Puerto Rican', 4:'Other Spanish/Hispanic', 5:'Asian alone, not-Hispanic', 6:'Native Hawaiian and Other Pacific Islander alone, not-Hispanic ', 7:'American Indian, Alaska native alone, not-Hispanic', 8:'Two or more races (not-Hispanic)'},'rec_race_a' : {1:'White alone', 2:'Black alone', 3:'American Indian/Alaska Native alone ', 4:'Asian alone', 5:'Native Hawaiian/Other Pacific Islander ', 6:'Two or more races'},'rec_race_c' : {1:'White alone', 2:'Black alone', 3:'American Indian/Alaska Native alone', 4:'Asian alone', 5:'Native Hawaiian/Other Pacific Islander', 6:'White and Black', 7:'White and American Indian and Alaska Native', 8:'White and Asian', 9:'White and Native Hawaiian/Other Pacific Islander alone', 10:'Black and American Indian/Alaska Native', 11:'Black and Asian', 12:'Black and Native Hawaiian/Other Pacific Islander', 13:'American Indian/Alaska Native and Asian', 14:'American Indian/Alaska Native and Native Hawaiian/Other Pacific Islander', 15:'Asian and Native Hawaiian/Other Pacific Islander', 16:'White and Black and American Indian/Alaska Native', 17:'White and Black and Asian', 18:'White and Black and Native Hawaiian/Other Pacific Islander', 19:'White and American Indian/Alaska Native and Asian', 20:'White and American Indian/Alaska Native and Native Hawaiian/Other Pacific Islander', 21:'White and Asian and Native Hawaiian/Other Pacific Islander', 22:'Black and American Indian/Alaska Native and Asian', 23:'Black and American Indian/Alaska Native and Native Hawaiian/Other Pacific Islander', 24:'Black and Asian and Native Hawaiian/Other Pacific Islander', 25:'American Indian/Alaska Native and Asian and Native Hawaiian/Other Pacific Islander', 26:'White and Black and American Indian/Alaska Native and Asian', 27:'White and Black and American Indian/Alaska Native and Native Hawaiian/Other Pacific Islander ', 28:'White and Black and Asian and Native Hawaiian/Other Pacific Islander', 29:'White and American Indian/Alaska Native and Asian and Native Hawaiian/Other Pacific Islander', 30:'Black and American Indian/Alaska Native and Asian and Native Hawaiian/Other Pacific Islander', 31:'White and Black and American Indian/Alaska Native and Asian and Native Hawaiian/Other Pacific Islander'},'rec62' : {1:'For exclusive use', 2:'Not for exclusive use', 4:'Some facilities in unit ', 5:'Lacking facilities in unit'},'rec64' : {1:'For exclusive use', 2:'Not for exclusive use', 4:'Some facilities in unit ', 5:'Facilities in building ', 6:'No facilities in building'},'rec54' : {1:'None', 2:'1 deficiency', 3:'2 deficiencies', 4:'3 deficiencies', 5:'4 deficiencies', 6:'5 deficiencies', 7:'Any item not reported'},'rec53' : {1:'None', 2:'1 deficiency', 3:'2 deficiencies', 4:'3 deficiencies', 5:'4 deficiencies', 6:'5 deficiencies', 7:'6 deficiencies', 8:'7 deficiencies', 9:'Any item not reported'},'tot_per' : {1:'1 person', 2:'2 person', 3:'3 person', 4:'4 person', 5:'5 person', 6:'6 person', 7:'7 person', 8:'8 person', 9:'9 person', 10:'10 person', 11:'11 person', 12:'12 person', 13:'13 person', 14:'14 person', 15:'15 person', 16:'16 person'},'rec39' : {1:'Household below 100% of income level ', 2:'Household between 100% to 124% of income level ', 3:'Household above or equal to 125% of income level'},'uf42a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf34a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf35a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf36a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf37a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf38a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf39a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'uf40a' : {0:'Amount shown contains no topcoded values ', 1:'Amount shown includes one or more topcoded values'},'cd' : {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06', 7:'07', 8:'08', 9:'09', 10:'10', 11:'11', 12:'12', 13:'13', 14:'14', 15:'15', 16:'16', 17:'17', 18:'18'},'rec8' : {1:'None' , 2:'One', 3:'Two' , 4:'Three' , 5:'Four', 6:'Five or more'},'rec7' : {1:'None' , 2:'One', 3:'Two' , 4:'Three' , 5:'Four', 6:'Five or more'},'flg_ag1' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag2' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag1' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag13' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag6' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag3' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag14' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag16' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag7' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag9' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag10' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag91' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag11' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'},'hflag12' : {1:'Allocated', 0:'Not allocated', 2:'Allocated default'}}

    #########replace no reply with np.nan for the categorical variables#########
    categorical_keys = categorical_dic.keys()
    df_categorical= df[categorical_keys]
    for key in categorical_keys:
        possible_values = categorical_dic[key].keys() #possible_values : list of known-value
        df_categorical[key][~(df_categorical[key].isin (possible_values))] = np.nan #replace anything outside known-value with NaN

    #########process numerical feature names#########
    numerical_dic= {'uf43':85 , 'uf2a': 12 ,'uf2b':12,'uf5':1385935,'uf6':5851008,'uf7':10813,'sc134':2014,'uf7a': 876,'uf8' :7721,'uf12':672,'uf13':694,'uf14' :816,'uf15':4587,'uf16':10388,'uf64' :696,'uf17':7999,'uf53':2014,'uf54':2014,'rec28':997,'uf42':9999997,'uf34':9999997,'uf35' :9999997,'uf36':9999997,'uf37': 9999997,'uf38':9999997,'uf39' : 9999997,'uf40':9999997}

    #########replace no reply with np.nan for the numerical variables#########
    numerical_keys = numerical_dic.keys()
    df_numerical = df[numerical_keys]
    for key in numerical_keys:
        possible_values = numerical_dic[key]#maximum meaningful value
        df_numerical[key][df_numerical[key] > possible_values] = np.nan #replace anything outside meaningful value with NaN

    #drop variables that have missing value only 
    def drop_missing_value(dataframe):
        all_missing = []
        for col in dataframe.columns:
            if len(dataframe[col].unique()) == 1 and np.isnan(dataframe[col].unique()[0]):
                all_missing.append(col)
        dataframe = dataframe.drop(all_missing,inplace=False,axis=1) #drop columns with all missing data
        return dataframe, all_missing


    df_binary, binary_all_missing = drop_missing_value(df_binary)
    df_categorical, categorical_all_missing = drop_missing_value(df_categorical)
    df_numerical, numerical_all_missing = drop_missing_value(df_numerical)
    #split numerical data into train and test
    cols = [col for col in df_numerical.columns if col not in ['uf17']]
    data_numerical = df_numerical[cols]
    target = df['uf17']
    X, y = data_numerical, target 
    X_train_nu, X_test_nu, y_train_nu, y_test_nu = train_test_split(X, y, random_state=0)
    #split binary data into train and test
    data_binary = df_binary
    X, y = data_binary, target
    X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(X, y, random_state=0)
    #split categorical data into train and test
    data_categorical = df_categorical
    X, y = data_categorical, target
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, random_state=0)
    #impute training data and transform test data based on training data specs 
    def impute(X_train, X_test, my_strategy):
        imp = Imputer(missing_values=np.nan, strategy=my_strategy).fit(X_train)
        X_train_imputed = imp.transform(X_train)
        X_train_imputed = pd.DataFrame(X_train_imputed, columns = X_train.columns)
        X_test_imputed = imp.transform(X_test)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns = X_test.columns)
        return X_train_imputed, X_test_imputed


    X_train_bi_imputed, X_test_bi_imputed =  impute(X_train_bi, X_test_bi, 'most_frequent')
    X_train_cat_imputed, X_test_cat_imputed =  impute(X_train_cat, X_test_cat, 'most_frequent')
    X_train_nu_imputed, X_test_nu_imputed =  impute(X_train_nu, X_test_nu, 'median')
    #concatenate binary, categorical, numerical into the final dataframe 
    X_train = pd.concat([X_train_bi_imputed, X_train_cat_imputed, X_train_nu_imputed], axis=1)
    X_test = pd.concat([X_test_bi_imputed, X_test_cat_imputed, X_test_nu_imputed], axis=1)
    y_train, y_test = y_train_nu, y_test_nu #y_test_nu == y_test_bi == y_test_cat, same as y_train_*
    #oneHot for the categorical data 
    categorical_all_missing = ['uf10', 'uf9', 'sc120', 'sc144', 'sc141']
    for c in categorical_dic.keys():
        if c in categorical_all_missing:
            continue
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    #dealing with train and test potential categorical feature inconsistency: say variable feature_6 shows up only in the training dataset but not test dataset, then it should be removed 
    train_col = X_train.columns
    test_col = X_test.columns
    missing_categorical_train_cols = []
    missing_categorical_test_cols = []
    if len(test_col) < len(train_col):
        for train in train_col:
            if train not in test_col:
                missing_categorical_test_cols.append(train)
    for test in test_col:
        if test not in train_col:
            missing_categorical_train_cols.append(test)
    X_train = X_train.drop(missing_categorical_test_cols,inplace=False,axis=1) #drop columns not exist in test data
    X_test = X_test.drop(missing_categorical_train_cols, inplace=False,axis=1) #drop columns not exist in train data
    return X_train, X_test, y_train, y_test

def feature_selection(X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso = LassoCV().fit(X_train_scaled, y_train)
    zero_coef_bol = lasso.coef_ == 0
    zero_coef = []
    for i in range(len(zero_coef_bol)):
        if zero_coef_bol[i] == True:
            zero_coef.append(X_train.columns[i])
    #remove features with zero lasso coef
    X_train = X_train.drop(zero_coef, inplace=False,axis=1)
    X_test = X_test.drop(zero_coef, inplace=False,axis=1)
    return X_train, X_test, y_train, y_test

def predict_rent(X_train, X_test, y_train, y_test):
    clf = Lasso(alpha=0.01425)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return X_test, y_test, predicted

def score_rent(y_test, predicted):
    return r2_score(y_test, predicted)

def main():
    X_train, X_test, y_train, y_test = process()
    #print (X_train.shape)
    X_train, X_test, y_train, y_test = feature_selection(X_train, X_test, y_train, y_test)
    #print (X_train.shape)
    X_test, y_test, predicted = predict_rent(X_train, X_test, y_train, y_test)
    Rs = score_rent(y_test, predicted)
    return Rs
    
    
if __name__ == "__main__":
    main()
