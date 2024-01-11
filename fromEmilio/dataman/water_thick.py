'''
Created on May 8, 2023

@author: politti
'''

from utils.utils import download_from_url
import os


def download_data(links_file, out_dir):
    
    downloads = []
    with open(links_file, 'r') as f:
        links = f.readlines()
        
        for link in links:
            downloaded = download_from_url(url=link, dest_folder=out_dir)
            
            if(os.path.isfile(downloaded)):
                downloads.append(downloaded)
                
    return downloads


if __name__ == '__main__':
    
    down_dir = '../data/total_water_storage'
    links_file = '../data/total_water_storage/opendap_links.txt'
    downloads = download_data(links_file=links_file, out_dir=down_dir)
    
    print('Process completed')
    
    
