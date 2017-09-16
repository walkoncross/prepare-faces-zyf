import os
import os.path

file_dir = "/disk2/du/face-asian"

fp = open("face_asian_list.txt",'w')

for root, dirs, files in os.walk(file_dir):
    for file in files:
        print file
        print os.path.join(root,file)
        fp.write('{}\n'.format(os.path.join(root,file)))
fp.close()
