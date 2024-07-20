from utils.data_loading_processing import *

google_drive_url=r"https://drive.google.com/file/d/1ZyuCNvVGmjxqTVuWwHa1Q_28qloNKfmN/view?usp=drive_link"
output_path=r"D:\projects_llm\Reviews\data"
# Download Data
#download_data_from_gdrive(google_drive_url,output_path)
read_part_file("D:\projects_llm\Reviews\data\datacgtolmy2.part",chunksize=1000)


