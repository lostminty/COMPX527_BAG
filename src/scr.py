




def load():
   data = []
   with open ("/src/output.txt", "r") as myfile:
       data=myfile.readlines()

   return data
