from classes.Model import Model
from math import floor

model_name = "model_PROVA"
songname = model_name+"_out.wav"

model = Model()
model.load(model_name)


print("GENERATING: ", songname)
model.generateFreewheel(songname, 1, 1000, floor(22050/2))

print()
print("SAVING embed data")
#model.embedInfoToFile("", filename=model_name+"_info.txt")