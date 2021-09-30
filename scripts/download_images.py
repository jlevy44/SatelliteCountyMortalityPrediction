import pandas as pd
import os
from os.path import join
import numpy as np
import os
import subprocess
from PIL import Image
import urllib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set(style='white')


def get_google_image(lat,lon,zoom,api_key,dimensions):
    url="""https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size={}x{}&maptype=satellite&key={}""".format(round(lat,6),round(lon,6),zoom,*dimensions,api_key)
    im=BytesIO(urllib.request.urlopen(url).read())
    #print(im)
    im = Image.open(im)
    return im


def main():
    image_dir='image_data/'
    w_h=400
    zoom=17
    key="API_KEY"
    country_img_df=pd.read_csv('data/image_locations_new.csv',index_col=0)
    c=0
    for county in country_img_df['County'].unique():
        dff=country_img_df[country_img_df['County']==county]
        for school,dfff in dff.groupby('School'):
            for i in range(dfff.shape[0]):
                os.makedirs(join(image_dir,county),exist_ok=True)
                outputimagefile=join(image_dir,county,'County_{}_School_{}_i_{}.png'.format(county,school,i))
                if not os.path.exists(outputimagefile):
                    print(outputimagefile)
                    im=get_google_image(dfff.iloc[i]['LAT'],dfff.iloc[i]['LON'],zoom,key,(w_h,w_h))
                    im.save(outputimagefile)
                    c+=1
                    time.sleep(max(1.,np.random.normal(3,2)))
                    if c>23000:
                        exit()




if __name__=='__main__':
    main()
