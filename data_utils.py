"""
Helpers for data handling
"""
import csv
import pandas as pd

DATAPATH = './data/'
IMAGE_FOLDER = '/IMG/'
CSV_IMAGE_FILE = 'driving_log.csv'

def loadData(folders): 
    '''
    Loads all samples data from the logs files. Specifying the different folder names where a driving_log.csv
    file may be found.  It changes the absolute paths of locations inside the csvs so that they fit on actual driving_log.csv 
    location.  This is done to allow moving a complete folder from one location to another.
    Args:
        folders:  list of folder names from main current project
    Returns:
        list containing all the driving_log csv rows with internal paths modified to final driving_log.csv file
    '''

    samples = []
    for dirname in folders:
        csvfilename = DATAPATH + dirname + '/' + CSV_IMAGE_FILE
        pref = DATAPATH + dirname + IMAGE_FOLDER
        with open(csvfilename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                #change file location and add to samples.
                line[0] = pref+line[0].split('\\')[-1]
                line[1] = pref+line[1].split('\\')[-1]
                line[2] = pref+line[2].split('\\')[-1]
                samples.append(line)
    
    return samples

def loadDataFrames(folders):
    """
    Loads all samples from disk (read loadData) and convert it to a data frame
    Args:
        folders: list of folder names from main current project
    Returns:
        Pandas dataframe containing all rows.
    """
    samples = loadData(folders)
    df = pd.DataFrame(samples)
    df.columns = ['imgcenter','imageleft', 'imageright',  'angle', 'throttle', 'breakpedal', 'speed']
    df.angle = df.angle.astype('float64')
    df.throttle = df.throttle.astype('float64')
    df.breakpedal = df.breakpedal.astype('float64')
    df.speed = df.speed.astype('float64')
    return df


def dataFrameWithLeftRight(df, correction=.4):
    '''
    From original data frame with columns 
    ['imgcenter','imageleft', 'imageright',  'angle', 'throttle', 'breakpedal', 'speed']
    we create another dataframe with left and rights too and columns
    ['imgfile', 'angle', 'throttle', 'speed']
    '''
    import scipy.signal

    #dfn = df[['imgcenter', 'angle', 'throttle', 'speed']]
    dfn = df.loc[:,('imgcenter', 'angle', 'throttle', 'speed')]
    dfn.columns=['imgfile', 'angle', 'throttle', 'speed']
   
    newrows = []
    
    dfn['angle'] = 1.3*scipy.signal.savgol_filter(df.angle,window_length=41,polyorder=2)


    for i,s in df.iterrows():
        # Add the left camera
        newangle = s.angle + correction
        if (newangle > 1.0):
            newangle = 1.0
        if (newangle <= 1.0):
            newrows.append({'imgfile': s.imageleft, 'angle':newangle, 'throttle':s.throttle, 'speed':s.speed} )
        # Add the right camera
        newangle = s.angle - correction
        if (newangle < -1.0):
            newangle = -1.0
        if (newangle >= -1.0):
            newrows.append({'imgfile':s.imageright, 'angle':newangle, 'throttle':s.throttle, 'speed':s.speed} )   
    
    return dfn.append(pd.DataFrame(newrows), ignore_index=True)



