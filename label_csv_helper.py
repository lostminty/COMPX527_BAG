# helper method to build CSV for use with datasets.py helper module
import csv,os,glob

LOCATION_OF_LABELS= "/home/lost/Downloads/archive/Labels/*.csv"
LOCATION_OF_VIDEOS = "/home/lost/Downloads/archive/DCSASS/*/*/*.mp4"
DESTINATION_FOR_NEW_LABELS='./__labels.csv'


mydict = dict()

file_paths_label_csv = glob.glob(LOCATION_OF_LABELS)
file_paths = glob.glob(LOCATION_OF_VIDEOS)

for file_path in file_paths_label_csv:
    with open(file_path) as infile:
        reader = csv.reader(infile)

        mydict.update({rows[0]:rows[1] for rows in reader})
        
output={path:(mydict[os.path.basename(path)[:-4]] if os.path.basename(path)[:-4] in mydict else "No Activity") for path in file_paths}
with open('./__labels.csv', 'w') as f:
    f.write("%s,%s\n"%("path","label"))
    for key in output.keys():
        f.write("%s,%s\n"%(key,output[key]))

#with open(DESTINATION_FOR_NEW_LABELS, 'w') as f:
#    for key in mydict.keys():
#        f.write("%s,%s\n"%(key,mydict[key]))