def findMissingFrames(df):
    missing = []
    prev = -1
    for x in df.frame_number.unique():
        x = int(x)
        if x != prev + 1:
            for i in range(prev+1, x):
                missing.append(i)
                
        prev = x
                
    return missing

def preProcessDF(df):

    # Converting the frames into a proper range
    for i in range(len(df)):
        df.iloc[i,0] -= 1
        df.iloc[i,0] = df.iloc[i,0]/10

    #

    return df