from predictor import predictor





data = []
with open ("output.txt", "r") as myfile:
       data=myfile.readlines()

print(predictor(data,"model.ckpt"))
