import os
def write_data(u,v,t):
   
   if not os.path.isdir(str(t)):
       os.makedirs(str(t))
   if not os.path.isdir(str(t)+'/U'):
       os.makedirs(str(t)+'/U')
   if not os.path.isdir(str(t)+'/V'):
       os.makedirs(str(t)+'/V')
   
   f = open(str(t)+'/U'+"/U.txt", "w")
   f.write(str(u))
   f.close()
   f = open(str(t)+'/V'+"/V.txt", "w")
   f.write(str(v))
   f.close()
   